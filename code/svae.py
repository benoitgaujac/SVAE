import numpy as np
import tensorflow as tf

import pdb

import distributions
import nn



## Initialize seeds
tf.set_random_seed(0)

######################################## Utils functions ########################################
# Initialization of discrete params
def init_cat(n_mixtures,dtype=tf.float32):
    """
    Initialize parameters of discrete distribution following dirichlet
    """
    discr_mean = tf.Variable(tf.ones(shape=[n_mixtures,1], dtype=dtype))# shape: [n_mixtures,1]
    # we use softmax to ensure the mean components sum to one
    soft_max = tf.nn.softmax(discr_mean,dim=0)
    return soft_max

# Initialization of mean and covariance matrix
def init_gaussian(n_mixtures,dim,dtype=tf.float32):
    """
    Initialize means and covariance matrix for the gaussian mixtures components
    """
    # mu
    mu = tf.Variable(tf.random_normal([n_mixtures,dim,1], mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,dim,1]
    # We have to enforce the covariance to be psd
    #Sigma
    # Sigma diagonal
    #diag = tf.Variable(tf.random_normal([n_mixtures,dim],mean=1.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,dim]
    diag = tf.Variable(tf.ones(shape=[n_mixtures,dim], dtype=dtype))# shape: [n_mixtures,1]
    sigma = tf.matrix_diag(tf.abs(diag))# shape: [n_mixtures,dim,dim]
    """
    # General psd sigma
    A = tf.Variable(tf.random_normal([n_mixtures,dim,dim],mean=1.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,dim,dim]
    eps = tf.Variable(tf.random_normal([n_mixtures,1,1], mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,1,1]
    sigma = 0.5 * (A + tf.transpose(A,perm=[0,2,1])) + tf.multiply(tf.eye(dim,batch_shape=[n_mixtures]),tf.abs(eps))# shape: [n_mixtures,dim,dim]
    """
    return tf.concat([mu,sigma],axis=-1) # shape: [n_mixtures,dim,1+dim]

def sample_gaussian(mean_params,dtype=tf.float32):
    """
    Sample guassian given mean and covariance
    """
    mu = tf.squeeze(mean_params[:,:,:,0])
    shape = mu.get_shape().as_list() + [1]
    sigma = mean_params[:,:,:,1:]
    chol = tf.cholesky(tf.squeeze(sigma))
    norm = tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=dtype)
    return tf.add(mu,tf.squeeze(tf.matmul(chol,norm),axis=-1))

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
class SVAE(object):
    """
    Stochastic Variation Autoencoder (SVAE).
    See "Composing graphical models with neural networks for structured representations and fast inference"
    by Johnson for more details.
    """
    def __init__(self, K=10, N=20, P=784, max_iter=20):
        """
        run inference and compute objective and optimizers for the SVAE algorithm
        network_architecture: architectures of the recognition and generator networks
        K: dim of z // number of components in mixture model
        N: x dimension // dimension of the latent gaussians
        P: y dimension // dimension of the observations
        nonlin: non linearity for the activation function of the networks
        """
        self.K=K
        self.N=N
        self.P=P
        self.max_iter = max_iter
        # Define the distributions of the model
        self._set_distributions()

    def _set_distributions(self):
        """
        Define the distributions of the model and initialize the parameters
        """
        # ExpFam used in the model. Handles the natparam and expstats computations
        self.labels = distributions.discrete()
        self.gaussian = distributions.gaussian()

    def _init_params(self,recog_archi,gener_archi):
        # Initialize nn parameters
        self.recognitionnet = nn.dense_net(recog_archi, "recog",data_type())
        self.generatornet = nn.dense_net(gener_archi, "gener",data_type())
        # Initialize parameters of the model \theta = (\pi, {(\mu_k, \Sigma_k)}_{k=1}^K)
        gauss_mean = init_gaussian(self.K,self.N,dtype=data_type())# shape: [n_mixtures,dim,1+dim]
        cat_mean = init_cat(self.K,dtype=data_type())# shape: [n_mixtures,1]
        return cat_mean,gauss_mean

    def init_label_stats(self):
        """
        Initialize label expected stats
        """
        stats = tf.ones(shape=[self.K,1], dtype=data_type())# shape: [n_mixtures,1]
        #stats = tf.random_normal([self.K,1], mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,1]
        soft_max = tf.nn.softmax(stats,dim=0)
        return soft_max

    def _build_recognition_net(self,input_):
        # Build recognition network and get potential
        recog_out = self.recognitionnet._build_network(input_)# shape: [batch,2N]
        # Outout of NN encode mu and the diag of sigma
        recog_mu = tf.reshape(recog_out[:,:self.N],[-1,1,self.N,1])# shape: [batch,1,N,1]
        # Sigma diagonal
        # we enforce it to be nsd
        recog_sigma = tf.expand_dims(tf.matrix_diag(-tf.abs(recog_out[:,self.N:])),axis=1)# shape: [batch,1,N,N]
        """
        # General psd sigma
        recog_A = devec(tf.expand_dims(recog_out[:,self.N+1:],axis=1),self.N,self.N)# shape: [batch,1,N,N]
        # we enforce it to be nsd
        eps = tf.reshape(tf.abs(recog_out[:,self.N]),[-1,1,1,1])# shape: [batch,1,1,1]
        recog_sigma = -(0.5 * (recog_A + tf.transpose(recog_A,perm=[0,1,3,2])) + tf.multiply(tf.eye(self.N,batch_shape=[1,1]),eps))# shape: [batch,1,N,N]
        """
        return tf.squeeze(vec(tf.concat([recog_mu,recog_sigma],axis=-1)),axis=1)# shape: [batch,N(N+1)]

    def _build_generator_net(self,input_):
        # Build generator network
        y_reconstr_mean = self.generatornet._build_network(input_)# shape: [batch,P*P]
        return y_reconstr_mean

    def _local_meanfield(self,node_potential,gaussian_global,label_global,label_stats):
        """
        Run fast inference.
        Compute local KL and optimal natural parameters of variational factors
        for the local meanfield using surrogate objective with recognition net.
        shape:
            - gaussian_global:  [batch,K,N,N+1]
            - label_global:     [batch,K,1]
            - label_stats:     [batch,K,1]
        """
        # compute local kl and natparams needed for sampling using partial optimizers
        gaussian_natparam, gaussian_stats, gaussian_kl = \
                self._gaussian_meanfield(gaussian_global, node_potential, label_stats)
        label_natparam, label_stats, label_kl = \
                self._label_meanfield(label_global, gaussian_global, gaussian_stats)

        kl = gaussian_kl+label_kl
        natparams = (label_natparam,gaussian_natparam)
        return kl, natparams

    def _meanfield_fixed_point(self,node_potential,gaussian_global,label_global,label_stats):
        """
        Compute partially optimizers of the surrogate objective.
        Return the expected labels stats need for the inference
        shape:
            - gaussian_global:  [batch,K,N,N+1]
            - label_global:     [batch,K,1]
            - label_stats:     [batch,K,1]
        """
        #label_stats_ = label_stats
        # block ascent to find partial optimizers
        for i in range(self.max_iter):
            gaussian_natparam, gaussian_stats, gaussian_kl = \
                self._gaussian_meanfield(gaussian_global, node_potential, label_stats)
            label_natparam, label_stats, label_kl = \
                self._label_meanfield(label_global, gaussian_global, gaussian_stats)
        return label_stats

    def _gaussian_meanfield(self,gaussian_global, node_potential, label_stats):
        """
        Compute gaussian variational factor
        shape:
            - gaussian_global:  [batch,K,N,N+1]
            - node_potential:   [batch,N*(N+1)]
            - label_stats:      [batch,K,1]
        """
        # compute the potential
        gaussian_global_flat = vec(gaussian_global)# shape: [batch,K,N(N+1)]
        global_potentials = tf.squeeze(tf.matmul(tf.transpose(label_stats,perm=[0,2,1]),gaussian_global_flat),axis=1)# shape: [batch,N(N+1)]
        # update gaussian natparams
        natparam = devec(tf.expand_dims(tf.add(node_potential,global_potentials),axis=1),self.N,self.N+1)# shape: [batch,1,N,N+1]
        self.natparam_test = natparam
        # get gaussian expected stats from updated natparams
        stats = self.gaussian.expectedstats(natparam)# shape: [batch,1,N,N+1]
        # compute gaussian kl
        stats_flat = vec(stats)
        dot_product = tf.reduce_sum(tf.multiply(stats_flat,node_potential),axis=-1,keep_dims=True)# shape: [batch,1]
        kl = dot_product - tf.squeeze(self.gaussian.logZ(natparam),axis=-1)
        return natparam, stats, kl

    def _label_meanfield(self,label_global, gaussian_global, gaussian_stats):
        """
        Compute label (discrete) variational factor
        shape:
            - label_global:     [batch,K,1]
            - gaussian_global:  [batch,K,N,N+1]
            - gaussian_stats:   [batch,1,N,N+1]
        """
        # compute the potential
        gaussian_global_flat = vec(gaussian_global)# shape: [batch,K,N(N+1)]
        gaussian_stats_flat = vec(gaussian_stats)# shape: [batch,1,N*(N+1)]
        global_potentials = tf.matmul(gaussian_global_flat,tf.transpose(gaussian_stats_flat,perm=[0,2,1]))# shape: [batch,K,1]
        # update labels natparams
        natparam = (global_potentials - self.gaussian.logZ(gaussian_global)) + label_global# shape: [batch,K,1]
        # get labels expected stats from updated natparams
        stats = self.labels.expectedstats(natparam)# shape: [batch,K,1]
        # Compute gaussian potential from updated labels expected stats
        gauss_potentials = tf.matmul(tf.transpose(stats,perm=[0,2,1]),gaussian_global_flat)# shape: [batch,1,N(N+1)]
        # compute labels kl
        kl = tf.squeeze(tf.matmul(gauss_potentials,tf.transpose(gaussian_stats_flat,perm=[0,2,1])),axis=-1)# shape: [batch,1]
        return natparam, stats, kl

    def _create_loss_optimizer(self,gaussian_global,label_global,label_stats_init,y,learning_rate,batch):
        """
        Compute the SVAE objective using reparametrization trick for likelyhood term
        shape:
            - gaussian_global:  [batch,K,N,N+1]
            - label_global:     [batch,K,1]
            - label_stats_init:     [batch,K,1]
        """
        # Build recognition network and compute node_potential
        node_potential = self._build_recognition_net(y)# shape: [batch,N(N+1)]
        # Compute partial optimizers for surrogate objective
        label_stats = self._meanfield_fixed_point(node_potential,gaussian_global,label_global,label_stats_init)# shape: [batch,K,1]
        # Compute local KL with partial otimized parameters
        local_KL, (labels_natparams,gaussian_natparams) = self._local_meanfield(node_potential,gaussian_global,label_global,label_stats)
        # Sample x from q, pass to generatornet
        gauss_mean = self.gaussian.natural_to_standard(gaussian_natparams)
        x = sample_gaussian(gauss_mean,data_type())# shape: [batch,N]
        # Build generator network and compute params of the obs variables from samples
        logits = self._build_generator_net(x)# shape: [batch,IMAGE_SIZE*IMAGE_SIZE]
        self.y_reconstr_mean = tf.sigmoid(logits)
        # Compute loglikeihood term
        loglikelihood = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
        # Compute SVAE objective
        self.SVAE_obj = tf.reduce_sum(loglikelihood - local_KL)# sum over batch
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.SVAE_obj,global_step=batch)
