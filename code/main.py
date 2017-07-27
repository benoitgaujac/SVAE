import gzip
import os
import sys
import time
import pdb

import numpy as np
import csv
from six.moves import urllib
from sklearn.utils import shuffle
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

import distributions
import nn
import svae

## Initialize seeds
np.random.seed(0)
tf.set_random_seed(0)

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '../../data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
BATCH_SIZE = 512
K = 15
N = 10
learning_rate = 0.001

num_epochs = 10


from optparse import OptionParser
parser = OptionParser()
parser.add_option('-m', '--model', action='store', dest='model',
    help="NN models in {gru1l32u, gru1l64u, gru1l128u, gru3l32u}")
parser.add_option('-s', '--mode', action='store', dest='mode',
    help="running mode in {train, test, inpainting} ")

######################################## Models architectures ########################################
#recognition_net = {"ninput":IMAGE_SIZE*IMAGE_SIZE,"nhidden_1":500,"nhidden_2":500,"noutput":N*(N+1)}
recognition_net = {"ninput":IMAGE_SIZE*IMAGE_SIZE,"nhidden_1":40,"nhidden_2":40,"noutput":2*N}
generator_net = {"ninput":N,"nhidden_1":40,"nhidden_2":40,"noutput":IMAGE_SIZE*IMAGE_SIZE}
nets_archi = {"recog":recognition_net,"gener":generator_net}

######################################## Data processing ########################################
def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0.0, 1.0].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_data():
    # download the data id needed
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    #train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    #test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    #train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    #test_labels = extract_labels(test_labels_filename, 10000)
    # Merge train and test_data
    data = np.concatenate((train_data,test_data))
    #labels = np.concatenate(train_labels,test_labels)
    return data

def get_batches(images, batch_size=BATCH_SIZE):
    batches = []
    X = shuffle(images)
    for i in range(int(X.shape[0]/batch_size)+1):
        if i<int(X.shape[0]/batch_size):
            X_batch = X[i * batch_size: (i + 1) * batch_size]
        else:
            X_batch = X[-batch_size:]
        batches.append(X_batch)
    return batches

def binarize(images, threshold=0.1):
    return (threshold < images).astype("float32")

######################################## Main ########################################
def main(nets_archi,data,mode_):
    data_size = data.shape[0]
    """
    # TODO
    # Create weights DST dir
    DST = create_DST(nn_model)
    """

    ###### Reset tf graph ######
    tf.reset_default_graph()
    start_time = time.time()
    print("\nPreparing variables and building model ...")

    ###### Create tf placeholder for obs variables ######
    y = tf.placeholder(dtype=data_type(), shape=(None, IMAGE_SIZE*IMAGE_SIZE))

    ###### Create instance SVAE ######
    recognition_net = nets_archi["recog"]
    generator_net = nets_archi["gener"]
    svae_ = svae.SVAE(K=K, N=N, P=IMAGE_SIZE*IMAGE_SIZE,
                                    learning_rate=learning_rate,
                                    batch_size=BATCH_SIZE)

    ###### Initialize parameters ######
    cat_mean,gauss_mean = svae_._initit_params(recognition_net,generator_net)
    #mu = tf.expand_dims(gauss_mean[:,:,0],axis=-1)
    #sigma = gauss_mean[:,:,1:]
    # We need to tile the natural parameters for each inputs in batch (inputs are iid)
    tile_shape = [BATCH_SIZE,1,1,1]
    gauss_mean_tiled = tf.tile(tf.expand_dims(gauss_mean,0),tile_shape)# shape: [batch,n_mixtures,dim,1+dim]
    cat_mean_tiled = tf.tile(tf.expand_dims(cat_mean,0),tile_shape[:-1])# shape: [batch,n_mixtures,1]
    # We convert the mean parameters to natural parameters
    gaussian_global = svae_.gaussian.standard_to_natural(gauss_mean_tiled)
    label_global = svae_.labels.standard_to_natural(cat_mean_tiled)
    # Initialize the labels expected stats for the block ascent algorithm
    labels_stats_init = tf.random_normal([BATCH_SIZE,K,1], mean=0.0, stddev=1.0, dtype=data_type())# shape: [batch,K,1]

    ###### Build loss and optimizer ######
    svae_._create_loss_optimizer(gaussian_global,label_global,labels_stats_init,y)


    """
    TODO
    ###### Create varaible for batch ######
    batch = tf.Variable(0, dtype=data_type())
    ###### CLearning rate decay ######
    learning_rate = tf.train.exponential_decay(
                    model_archi["init_learning_rate"],  # Base learning rate.
                    batch * BATCH_SIZE,                 # Current index into the dataset.
                    5*train_size,                       # Decay step.
                    0.90,                               # Decay rate.
                    staircase=True)
    """
    ###### Initializer ######
    init = tf.global_variables_initializer()
    ###### Saver ######
    saver = tf.train.Saver()
    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        # Training
        if mode_!="test":
            """
            # Opening csv file
            csv_path = "../../trained_models/part2/Perf/Training_"
            csvfileTrain = open(csv_path + str(nn_model) + ".csv", 'w')
            Trainwriter = csv.writer(csvfileTrain, delimiter=';',)
            Trainwriter.writerow(['Num Epoch', 'Time', 'Training loss'])
            """
            #if not tf.gfile.Exists(DST+".ckpt.meta") or not from_pretrained_weights:
            sess.run(tf.global_variables_initializer())
            #sess.run(tf.global_variables_initializer(), feed_dict={phase_train: True})
            """
            TODO: reload trained weights
            else:
            # Load pre trained model if exist
            saver.restore(sess, DST+".ckpt")
            # Reinitialize learning rate
            tf.variables_initializer([batch,]).run()
            learning_rate = tf.train.exponential_decay(
                            model_archi["init_learning_rate"],  # Base learning rate.
                            batch * BATCH_SIZE,                 # Current index into the dataset.
                            5*train_size,                       # Decay step.
                            0.90,                               # Decay rate.
                            staircase=True)
            """
            # initialize performance indicators
            l_history = [-100000.0,]
            best_l = -10000000000.0
            #training loop
            print("\nStart training ...")
            for epoch in range(num_epochs):
                start_time = time.time()
                train_l = 0.0
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                batches = get_batches(data, BATCH_SIZE)
                for batch in batches:
                    # Run the optimizer to update weights and get loss.
                    _,l= sess.run([svae_.optimizer,svae_.SVAE_obj], feed_dict={y: batch})
                    # Update average loss and accuracy
                    train_l += l / len(batches)
                if train_l>best_l:
                    best_l = train_l
                #Test for ploting images
                piy= sess.run(svae_.y_reconstr_mean, feed_dict={y: data[:BATCH_SIZE]})
                init = np.reshape(data[:BATCH_SIZE],[-1,IMAGE_SIZE,IMAGE_SIZE])*255.0 #shape: [batch, 28, 28]
                init = init.astype("int32")
                gene = np.reshape(piy,[-1,IMAGE_SIZE,IMAGE_SIZE])*255.0 #shape: [batch, 28, 28]
                gene =  gene.astype("int32")
                for i in range(10):
                    fig = plt.figure()
                    # original image
                    plt.subplot(1,2,1)
                    plt.imshow(init[i], cmap="gray", interpolation=None)
                    plt.axis("on")
                    # generated image
                    plt.subplot(1,2,2)
                    plt.imshow(gene[i], cmap="gray", interpolation=None)
                    plt.axis("on")
                    fig.savefig("im" + str(i) + ".png")
                    plt.close()

                # Print info for previous epoch
                print("Epoch {} done, took {:.2f}s, learning rate: {:.2f}e-3".format(epoch,time.time()-start_time,learning_rate))
                print("Epoch loss: {:.4f}, Best train loss: {:.4f}".format(train_l,best_l))
                """
                # Writing csv file with results and saving models
                Trainwriter.writerow([epoch + 1, time.time() - start_time, train_loss])
                """
        """
        # Testing
        csv_path = "../../trained_models/part2/Perf/test_"
        csvfileTest = open(csv_path + str(nn_model) + ".csv", 'w')
        Testwriter = csv.writer(csvfileTest, delimiter=';',)
        Testwriter.writerow(['Test loss'])
        if not tf.gfile.Exists(DST+".ckpt.meta"):
            raise Exception("no weights given")
        saver.restore(sess, DST+".ckpt")
        # Compute and print results once training is done
        test_loss, test_acc = 0.0, 0.0
        test_Batches = get_batches(test_data, BATCH_SIZE_EVAL)
        for test_batch in test_Batches:
            tst_loss, tst_pred = sess.run([loss, prediction], feed_dict={
                                                    data_node: test_batch,
                                                    phase_train: False})
            test_loss += tst_loss / len(test_Batches)
            test_acc += accuracy_logistic(tst_pred,test_batch[:,1:]) / len(test_Batches)
        print("\nTesting after {} epochs.".format(num_epochs))
        print("Test loss: {:.4f}, Test acc: {:.2f}%".format(test_loss,test_acc*100))
        logging.info("\nTest loss: {:.4f}, Test acc: {:.2f}%".format(test_loss,test_acc*100))
        Testwriter.writerow([test_loss])
        """

if __name__ == '__main__':
    ###### Load and get data ######
    data = get_data()
    data = data[:5000]
    # Reshape data
    data = np.reshape(data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    # Convert to binary
    print("Converting data to binary")
    data = binarize(data)
    main(nets_archi,data,"traning")

    """
    options, arguments = parser.parse_args(sys.argv)
    if options.model not in models.keys():
        raise Exception("Invalide model name")
    else:
        if options.mode=="test" or options.mode=="train":
            # Shuffle train data
            np.random.shuffle(train_data)
            main(models[options.model],train_data, validation_data, test_data, options.mode)
        elif options.mode=="inpainting":
            # pixel in-painting
            cache_data, idx = get_cache_data_set(test_data,nsample=nsample)
            inpainting.in_painting(models[options.model],test_data[idx],cache_data)
        else:
            raise Exception("Invalide mode")
    """
