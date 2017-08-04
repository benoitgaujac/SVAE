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


## Initialize seeds
np.random.seed(0)
tf.set_random_seed(0)

IMAGE_SIZE = 28
BATCH_SIZE = 4
K = 4
N = 15
learning_rate_init = 0.002
niter = 80
num_epochs = 100
nexamples = 4
nsamples = 3

"""
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-m', '--model', action='store', dest='model',
    help="NN models in {gru1l32u, gru1l64u, gru1l128u, gru3l32u}")
parser.add_option('-s', '--mode', action='store', dest='mode',
    help="running mode in {train, test, inpainting} ")
"""

######################################## Models architectures ########################################
#recognition_net = {"ninput":IMAGE_SIZE*IMAGE_SIZE,"nhidden_1":500,"nhidden_2":500,"noutput":N*(N+1)+1}
recognition_net = {"ninput":IMAGE_SIZE*IMAGE_SIZE,"nhidden_1":512,"nhidden_2":512,"noutput":2*N}
generator_net = {"ninput":N,"nhidden_1":512,"nhidden_2":512,"noutput":IMAGE_SIZE*IMAGE_SIZE}
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
    img_to_plot = np.reshape(original,[-1,IMAGE_SIZE,IMAGE_SIZE])*255.0# shape: [nexamples, 28, 28]
    img_to_plot = img_to_plot.astype("int32")
    mean = np.reshape(bernouilli_mean,[-1,IMAGE_SIZE,IMAGE_SIZE])*255.0# shape: [nexamples, 28, 28]
    mean =  mean.astype("int32")
    bernoulli = tf.contrib.distributions.Bernoulli(probs=tf.tile(tf.expand_dims(bernouilli_mean,axis=1),[1,nsamples,1]), dtype=tf.float32)# shape: [nexamples,nsamples,28*28]
    bernoulli_samples = tf.reshape(bernoulli.sample(),[-1,nsamples,IMAGE_SIZE,IMAGE_SIZE])*255.0# shape: [nexamples,nsamples,28,28]
    bernoulli_samples = bernoulli_samples.eval().astype("int32")
    """
    for i in range(nexamples):
        fig = plt.figure()
        for j in range(2+nsamples):
            plt.subplot(1,2+nsamples,j+1)
            if j==0:
                plt.title("Original", fontsize=10)
                plt.imshow(img_to_plot[i], cmap="gray", interpolation=None)
            elif j==1:
                plt.title("Bernouilli mean", fontsize=10)
                plt.imshow(mean[i], cmap="gray", interpolation=None)
            else:
                plt.title("Sample " + str(j-1), fontsize=10)
                plt.imshow(bernoulli_samples[i,j-2], cmap="gray", interpolation=None)
            plt.axis("on")
        if not tf.gfile.Exists(DST):
            os.makedirs(DST)
        file_name = os.path.join(DST, "Example" + str(i) + ".png")
        fig.savefig(file_name)
        plt.close()
    """
    fig = plt.figure()
    for i in range(nexamples):
        for j in range(2+nsamples):
            plt.subplot(2+nsamples,nexamples,j*nexamples+i+1)
            plt.axis("off")
            if j==0:
                #plt.title("Original", fontsize=10)
                plt.imshow(img_to_plot[i], cmap="gray", interpolation=None)
            elif j==1:
                #plt.title("Bernouilli mean", fontsize=10)
                plt.imshow(mean[i], cmap="gray", interpolation=None)
            else:
                #plt.title("Sample " + str(j-1), fontsize=10)
                plt.imshow(bernoulli_samples[i,j-2], cmap="gray", interpolation=None)
    if not tf.gfile.Exists(DST):
        os.makedirs(DST)
    file_name = os.path.join(DST, "Example.png")
    fig.savefig(file_name)
    plt.close()


def save_gene(bernouilli_mean, DST):
    mean = np.reshape(bernouilli_mean,[-1,IMAGE_SIZE,IMAGE_SIZE])*255.0# shape: [K, 28, 28]
    mean =  mean.astype("int32")
    bernoulli = tf.contrib.distributions.Bernoulli(probs=tf.tile(tf.expand_dims(bernouilli_mean,axis=1),[1,nsamples,1]), dtype=tf.float32)# shape: [K,nsamples,28*28]
    bernoulli_samples = tf.reshape(bernoulli.sample(),[-1,nsamples,IMAGE_SIZE,IMAGE_SIZE])*255.0# shape: [K,nsamples,28,28]
    bernoulli_samples = bernoulli_samples.eval().astype("int32")
    for i in range(K):
        fig = plt.figure()
        for j in range(nsamples):
            plt.subplot(1,nsamples,j+1)
            plt.title("Sample " + str(j+1), fontsize=10)
            plt.imshow(bernoulli_samples[i,j], cmap="gray", interpolation=None)
            plt.axis("on")
        if not tf.gfile.Exists(DST):
            os.makedirs(DST)
        file_name = os.path.join(DST, "mixture" + str(i) + ".png")
        fig.savefig(file_name)
        plt.close()

######################################## Main ########################################
def main(nets_archi,data,mode_,name="test"):
    data_size = data.shape[0]
    # Create weights DST dir
    DST = create_DST(name)

    ###### Reset tf graph ######
    tf.reset_default_graph()
    start_time = time.time()
    print("\nPreparing variables and building model ...")

    ###### Create tf placeholder for obs variables ######
    y = tf.placeholder(dtype=data_type(), shape=(None, IMAGE_SIZE*IMAGE_SIZE))

    ###### Create varaible for batch ######
    batch = tf.Variable(0, dtype=data_type())
    ###### CLearning rate decay ######
    learning_rate = tf.train.exponential_decay(
                    learning_rate_init,     # Base learning rate.
                    batch * BATCH_SIZE,     # Current index into the dataset.
                    20*data_size,              # Decay step.
                    0.99,                   # Decay rate.
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
    labels_stats_init,label_global_mean,gauss_global_mean = svae_._init_params()
    # We need to tile the natural parameters for each inputs in batch (inputs are iid)
    tile_shape = [BATCH_SIZE,1,1,1]
    gauss_global_mean_tiled = tf.tile(tf.expand_dims(gauss_global_mean,0),tile_shape)# shape: [batch,n_mixtures,dim,1+dim]
    label_global_mean_tiled = tf.tile(tf.expand_dims(label_global_mean,0),tile_shape[:-1])# shape: [batch,n_mixtures,1]
    labels_stats_init_tiled = tf.tile(tf.expand_dims(labels_stats_init,0),tile_shape[:-1])# shape: [batch,K,1]
    # We convert the global mean parameters to global natural parameters
    gaussian_global = svae_.gaussian.standard_to_natural(gauss_global_mean_tiled)
    label_global = svae_.labels.standard_to_natural(label_global_mean_tiled)

    ###### Build loss and optimizer ######
    svae_._create_loss_optimizer(gaussian_global,label_global,
                                        labels_stats_init_tiled,
                                        y,
                                        learning_rate,
                                        batch)
    ###### Build generator ######
    svae_._generate(tf.expand_dims(gauss_global_mean,0))

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
            Trainwriter.writerow(['Num Epoch', 'objective'])
            sess.run(tf.global_variables_initializer())
            # initialize performance indicators
            best_l = -10000000000.0
            #training loop
            print("\nStart training ...")
            for epoch in range(num_epochs):
                start_time = time.time()
                train_l = 0.0
                print("")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                batches = data_processing.get_batches(data, BATCH_SIZE)
                for batch in batches:
                    _,l,recon_mean,gener_mean,lr = sess.run([svae_.optimizer,svae_.SVAE_obj,svae_.y_reconstr_mean,svae_.y_generate_mean,learning_rate], feed_dict={y: batch})
                    # Update average loss
                    train_l += l/len(batches)
                if train_l>best_l:
                    best_l = train_l
                # Print info for previous epoch
                print("Epoch {} done, took {:.2f}s, learning rate: {:10.2e}".format(epoch,time.time()-start_time,lr))
                print("Epoch loss: {:10.2e}, Best train loss: {:10.2e}".format(train_l,best_l))
                """
                # Writing csv file with results and saving models
                Trainwriter.writerow([epoch + 1, train_l])
                saver.save(sess,DST)
                """
                save_reconstruct(batch[:nexamples], recon_mean[:nexamples], "./reconstruct")
                save_gene(gener_mean, "./generate")

        if mode_=="reconstruct":
            #Test for ploting images
            if not tf.gfile.Exists(DST+".ckpt.meta"):
                raise Exception("no weights given")
            saver.restore(sess, DST+".ckpt")
            img = data[np.random.randint(0, high=data_size, size=BATCH_SIZE)]
            bernouilli_mean= sess.run(svae_.y_reconstr_mean, feed_dict={y: img})
            save_reconstruct(img[:nexamples], bernouilli_mean[:nexamples], "./reconstruct")

        if mode_=="generate":
            #Test for ploting images
            if not tf.gfile.Exists(DST+".ckpt.meta"):
                raise Exception("no weights given")
            saver.restore(sess, DST+".ckpt")
            gaussian_mean= sess.run(gauss_global_mean, feed_dict={})
            bernouilli_mean= sess.run(svae_.y_generate_mean, feed_dict={gaussian_mean: gaussian_mean})
            save_gene(bernouilli_mean, "./generate")


if __name__ == '__main__':
    ###### Load and get data ######
    data = shuffle(data_processing.get_data())
    data = data[:1*BATCH_SIZE]
    #data = data[:10000]
    # Reshape data
    data = np.reshape(data,[-1,IMAGE_SIZE*IMAGE_SIZE])
    # Convert to binary
    print("Converting data to binary")
    data = data_processing.binarize(data)
    main(nets_archi,data,"training")

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
