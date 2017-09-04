import os
import sys
import time
import pdb

import numpy as np
import csv
from sklearn.utils import shuffle

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from math import pi

import data_processing
import distributions
import nn
import svae


# Initialize seeds
tf.set_random_seed(time.localtime())
np.random.seed(time.localtime())

IMAGE_SIZE = 28
BATCH_SIZE = 2048
K = 10
N = 15
learning_rate_init = 0.004
niter = 30
num_epochs = 200
nexamples = 5

"""
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-m', '--model', action='store', dest='model',
    help="NN models in {gru1l32u, gru1l64u, gru1l128u, gru3l32u}")
parser.add_option('-s', '--mode', action='store', dest='mode',
    help="running mode in {train, test, inpainting} ")
"""

######################################## Models architectures ########################################
#recognition_net = {"ninput":IMAGE_SIZE*IMAGE_SIZE,"nhidden_1":512,"nhidden_2":512,"nhidden_3":512,"noutput":N*(N+1)}
recognition_net = {"ninput":IMAGE_SIZE,"num_filters":32,"size_filters_1":5,"size_filters_2":3,"fc":128,"noutput":2*N}
generator_net = {"ninput":N,"fc":128,"num_filters":32,"size_filters_1":3,"size_filters_2":5,"noutput":IMAGE_SIZE}
nets_archi = {"recog":recognition_net,"gener":generator_net}

######################################## Utils ########################################
def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32

def create_DST(name):
    DIR = "./trained_models"
    if not tf.gfile.Exists(DIR):
        os.makedirs(DIR)
    PATH = os.path.join(DIR,name)
    return PATH

def save_reconstruct(original, bernouilli_mean, DST):
    orig = np.reshape(original,[-1,IMAGE_SIZE,IMAGE_SIZE])*255.0# shape: [nexamples, 28, 28]
    orig = orig.astype("int32")
    mean = np.reshape(bernouilli_mean,[-1,IMAGE_SIZE,IMAGE_SIZE])*255.0# shape: [nexamples, 28, 28]
    mean =  mean.astype("int32")
    fig = plt.figure()
    for i in range(2*nexamples):
        plt.subplot(2,nexamples,i+1)
        plt.axis("off")
        if i<nexamples:
            plt.title("Original", fontsize=10)
            plt.imshow(orig[i], cmap="gray", interpolation=None)
        else :
            plt.title("Reconstruct", fontsize=10)
            plt.imshow(mean[i-nexamples], cmap="gray", interpolation=None)
    if not tf.gfile.Exists(DST):
        os.makedirs(DST)
    file_name = os.path.join(DST, "reconstruct_test.png")
    fig.savefig(file_name)
    plt.close()

def save_gene(bernouilli_mean, DST):
    mean = bernouilli_mean*255.0# shape: [K, nexample, 28, 28]
    mean =  mean.astype("int32")# shape: [K, nexample, 28, 28]
    fig = plt.figure()
    for i in range(K):
        for j in range(nexamples):
            plt.subplot(K,nexamples,nexamples*i+j+1)
            plt.axis("off")
            plt.imshow(mean[i,j], cmap="gray", interpolation=None)
    if not tf.gfile.Exists(DST):
        os.makedirs(DST)
    file_name = os.path.join(DST, "mixtures.png")
    fig.savefig(file_name)
    plt.close()

######################################## Main ########################################
def main(nets_archi,train_data,test_data,mode_,name="test"):
    # Preprocessing data
    data_size = train_data.shape[0]
    # Create weights DST dir
    DST = create_DST(name)

    ###### Reset tf graph ######
    tf.reset_default_graph()
    start_time = time.time()
    print("\nPreparing variables and building model ...")

    ###### Create tf placeholder for obs variables ######
    y = tf.placeholder(dtype=data_type(), shape=(None, IMAGE_SIZE,IMAGE_SIZE,1))
    normal_mean = tf.placeholder(dtype=data_type(), shape=(K,N,N+1))

    ###### Create varaible for batch ######
    batch = tf.Variable(0, dtype=data_type())
    ###### CLearning rate decay ######
    learning_rate = tf.train.exponential_decay(
                    learning_rate_init,     # Base learning rate.
                    batch * BATCH_SIZE,     # Current index into the dataset.
                    10*data_size,              # Decay step.
                    0.98,                   # Decay rate.
                    staircase=True)

    ###### Create instance SVAE ######
    recognition_net = nets_archi["recog"]
    generator_net = nets_archi["gener"]
    svae_ = svae.SVAE(recog_archi=recognition_net,  # architecture of the recognition network
                        gener_archi=generator_net,  # architecture of the generative network
                        K=K,                        # dim of the discrete latents z
                        N=N,                        # dim of the gaussian latents x
                        P=IMAGE_SIZE*IMAGE_SIZE,    # dim of the obs variables y
                        max_iter=niter)             # number of iterations in the coordinate block ascent

    ###### Initialize parameters ######
    labels_stats_init_tiled,label_global_mean,gauss_global_mean = svae_._init_params()
    # We need to tile the natural parameters for each inputs in batch (inputs are iid)
    tile_shape = [BATCH_SIZE,1,1,1]
    gauss_global_mean_tiled = tf.tile(tf.expand_dims(gauss_global_mean,0),tile_shape)# shape: [batch,n_mixtures,dim,1+dim]
    label_global_mean_tiled = tf.tile(tf.expand_dims(label_global_mean,0),tile_shape[:-1])# shape: [batch,n_mixtures,1]
    # We convert the global mean parameters to global natural parameters
    gaussian_global = svae_.gaussian.standard_to_natural(gauss_global_mean_tiled)
    label_global = svae_.labels.standard_to_natural(label_global_mean_tiled)

    ###### Build loss and optimizer ######
    svae_._create_loss_optimizer(gaussian_global,
                                    label_global,
                                    labels_stats_init_tiled,
                                    y,
                                    learning_rate,
                                    batch)

    ###### Build generator ######
    svae_._generate(tf.tile(tf.expand_dims(normal_mean,0),[nexamples,1,1,1]))

    ###### Initializer ######
    init = tf.global_variables_initializer()
    ###### Saver ######
    saver = tf.train.Saver()
    ###### Create a local session to run the training ######
    with tf.Session() as sess:
        # Training
        if mode_=="training":
            # Opening csv file
            csv_path = "./Perf"
            if not tf.gfile.Exists(csv_path):
                os.makedirs(csv_path)
            csvfileTrain = open(os.path.join(csv_path,name) + ".csv", 'w')
            Trainwriter = csv.writer(csvfileTrain, delimiter=';',)
            Trainwriter.writerow(['Num Epoch', 'train loss', 'test_loss'])

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # initialize performance indicators
            best_l = -10000000000.0

            #training loop
            print("\nStart training ...")
            for epoch in range(num_epochs):
                start_time = time.time()
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # Training loop
                train_l = 0.0
                batches = data_processing.get_batches(train_data, BATCH_SIZE)
                for i,batch in enumerate(batches):
                    _,l,lr =sess.run([svae_.optimizer,svae_.SVAE_obj,
                                                    learning_rate],
                                                    feed_dict={y: batch})
                    # Update average loss
                    train_l += l/len(batches)

                # Testing loop
                test_l = 0.0
                batches = data_processing.get_batches(test_data, BATCH_SIZE)
                for i,batch in enumerate(batches):
                    l =sess.run(svae_.SVAE_obj,feed_dict={y: batch})
                    # Update average loss
                    test_l += l/len(batches)

                # Update best perf and save model
                if test_l>best_l:
                    best_l = test_l
                    if epoch>20:
                        saver.save(sess,DST)
                        print("model saved.")
                # Print info for previous epoch
                print("Epoch {} done, took {:.2f}s, learning rate: {:10.2e}".format(epoch,time.time()-start_time,lr))
                print("Train loss: {:.3f}, Test loss: {:.3f},Best test loss: {:.3f}".format(train_l,test_l,best_l))

                # Writing csv file with results and saving models
                Trainwriter.writerow([epoch + 1, train_l, test_l])

        if mode_=="reconstruct":
            #Plot reconstruction mean
            if not tf.gfile.Exists(DST+".meta"):
                raise Exception("no weights given")
            saver.restore(sess, DST)
            img = test_data[np.random.randint(0, high=test_data.shape[0], size=BATCH_SIZE)]
            bernouilli_mean= sess.run(svae_.y_reconstr_mean, feed_dict={y: img})
            save_reconstruct(img[:nexamples], bernouilli_mean[:nexamples], "./reconstruct")

        if mode_=="generate":
            #Test for ploting images
            if not tf.gfile.Exists(DST+".meta"):
                raise Exception("no weights given")
            saver.restore(sess, DST)
            gaussian_mean= sess.run(gauss_global_mean, feed_dict={})
            #pdb.set_trace()
            bernouilli_mean = sess.run(svae_.y_generate_mean, feed_dict={normal_mean: gaussian_mean})
            bernouilli_mean = np.transpose(np.reshape(bernouilli_mean,(nexamples,K,IMAGE_SIZE,IMAGE_SIZE)),(1,0,2,3))
            save_gene(bernouilli_mean, "./generate")


if __name__ == '__main__':
    ###### Load and get data ######
    train_data,test_data = data_processing.get_data("MNIST")
    #train_data = train_data[:40000]
    """
    data = shuffle(data_processing.get_data())
    #data = data[:1*BATCH_SIZE]
    data = data[:10000]
    """
    # Convert to binary
    print("Converting data to binary")
    train_data = data_processing.binarize(train_data)
    test_data = data_processing.binarize(test_data)
    main(nets_archi,train_data,test_data,"training","full")

    """
    TODO
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
