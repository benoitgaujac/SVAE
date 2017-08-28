import numpy as np
import tensorflow as tf

import pdb

import nn
from main import BATCH_SIZE



## Initialize seeds
tf.set_random_seed(0)

######################################## Utils functions ########################################
def sample_gaussian(mu_,sigma_,dtype=tf.float32,batch=BATCH_SIZE):
    """
    Sample guassian given mean and covariance
    """
    mu = tf.expand_dims(mu_,axis=-1)# shape: [batch,N,1]
    chol = tf.cholesky(sigma_)# shape: [batch,N,N]
    N = mu.get_shape().as_list()[1]
    norm = tf.random_normal([batch,N,1], mean=0.0, stddev=1.0, dtype=dtype)# shape: [batch,N,1]
    return tf.add(mu,tf.matmul(chol,norm))# shape: [batch,N,1]

def vec(x):
    """
    vec operator stacking columns of x
    shape:
        - x:  [batch,K,N,M]
    return vec of the matrix formed by the inner-most 2 dim, shape: [batch,K,N*M]
    """
    [K,N,M] = x.get_shape().as_list()[-3:]
    return tf.reshape(tf.transpose(x,perm=[0,1,3,2]),[-1,K,N*M])

def devec(x,N,M):
    """
    perfom inverse of vec operator
    shape:
        - x:  [batch,K,N*M]
    return tensor whose the inner-most 2 dim are formed by concatening
    the columns over the last dimension, shape: [batch,K,N*M]
    """
    K = x.get_shape().as_list()[-2]
    return tf.transpose(tf.reshape(x,[-1,K,M,N]),perm=[0,1,3,2])

def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    return tf.float32


######################################## SVAE ########################################
class VAE(object):
    """
    Variation Autoencoder (VAE).
    """
    def __init__(self,recog_archi, gener_archi, N=20, P=784):
        """
        run inference and compute objective and optimizers for the VAE algorithm
        network_architecture: architectures of the recognition and generator networks
        N: x dimension // dimension of the latent gaussians
        P: y dimension // dimension of the observations
        """
        self.N=N
        self.P=P
        # Initialize weights of the NNs
        self._init_weights(recog_archi,gener_archi)

    def _init_weights(self,recog_archi,gener_archi):
        # Initialize nn parameters
        self.recognitionnet = nn.conv_net_4l(recog_archi, "recog",data_type())
        self.generatornet = nn.deconv_net_4l(gener_archi, "gener",data_type())

    def _build_recognition_net(self,input_):
        # Foward pass for recognition network and get potential
        recog_out = self.recognitionnet._build_network(input_)# shape: [batch,2N]
        # Outout of NN encode mu and the diag of sigma
        recog_mu = recog_out[:,:self.N]# shape: [batch,N]
        # Diagonal sigma
        recog_sigma = tf.matrix_diag(tf.exp(recog_out[:,self.N:]))# shape: [batch,N,N]
        return recog_mu,recog_sigma

    def _build_generator_net(self,input_):
        # Foward pass for generator network
        y_reconstr_mean = self.generatornet._build_network(input_)# shape: [batch,IMAGE_SIZE,IMAGE_SIZE,1]
        return y_reconstr_mean

    def _local_KL(self,recog_mu,recog_sigma):
        # Compute local KL
        recog_sigma_diag = tf.matrix_diag_part(recog_sigma)
        kl = 1 + tf.log(tf.square(recog_sigma_diag)) - tf.square(recog_mu) - tf.square(recog_sigma_diag)
        return 0.5*tf.reduce_sum(kl,axis=-1)#shape: [batch,1]

    def _create_loss_optimizer(self,y,learning_rate,batch):
        """
        Compute the VAE objective using reparametrization trick for the expected likelyhood term
        """
        # Build recognition network and compute node_potential
        recog_mu, recog_sigma = self._build_recognition_net(y)# shape: [batch,N(N+1)]
        # Compute local KL
        self.local_KL = self._local_KL(recog_mu,recog_sigma)
        # Sample x from q, pass to generatornet
        x = sample_gaussian(recog_mu,recog_sigma,data_type())# shape: [batch,N,1]
        # Build generator network and compute params of the obs variables from samples
        logits = self._build_generator_net(tf.squeeze(x))# shape: [batch,IMAGE_SIZE,IMAGE_SIZE,1]
        self.y_reconstr_mean = tf.nn.sigmoid(logits)# shape: [batch,IMAGE_SIZE,IMAGE_SIZE,1]
        # Compute loglikeihood term
        self.loglikelihood = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits),axis=[1,2,3])
        # Compute VAE objective
        self.VAE_obj = tf.reduce_mean(self.loglikelihood + self.local_KL)# average over batch
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-self.VAE_obj,global_step=batch)

    def _generate(self,nexamples):
        """
        Generate sample from centered Normal
        """
        mu = tf.zeros([nexamples,self.N],data_type())
        sigma = tf.eye(self.N,batch_shape=[nexamples],dtype=data_type())
        x = sample_gaussian(mu,sigma,dtype=data_type(),batch=nexamples)# shape: [nexamples,N,1]
        # Build generator network and compute params of the obs variables from samples
        logits = self._build_generator_net(tf.reshape(x,[-1,self.N]))# shape: [nexamples,IMAGE_SIZE,IMAGE_SIZE,1]
        self.y_generate_mean = tf.sigmoid(logits)# shape: [nexamples*K,IMAGE_SIZE,IMAGE_SIZE,1]
