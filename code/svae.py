import numpy as np
import tensorflow as tf

import pdb

import distributions
import nn



## Initialize seeds
tf.set_random_seed(0)

######################################## Utils functions ########################################
# Initialization of discrete params
def init_cat(n_mixtures,params="global",dtype=tf.float32):
    """
    Initialize parameters of discrete distribution following dirichlet
    """
    if params=="global":
        discr_mean = tf.Variable(tf.random_normal(shape=[n_mixtures,1], mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,1]
    elif params=="variational":
        #discr_mean = tf.ones(shape=[n_mixtures,1], dtype=dtype)# shape: [n_mixtures,1]
        discr_mean = tf.random_normal(shape=[n_mixtures,1], mean=0.0, stddev=1.0, dtype=dtype)# shape: [n_mixtures,1]
    else:
        raise Exception("Wrong type of parameters. No parameters initialized")
    # Softmax to ensure sum to 1
    logits = tf.nn.softmax(discr_mean,dim=0)
    return logits

# Initialization of mean and covariance matrix
def init_gaussian(n_mixtures,dim,dtype=tf.float32):
    """
    Initialize means and covariance matrix for the gaussian mixtures components
    """
    mu = tf.Variable(tf.random_normal([n_mixtures,dim,1], mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,dim,1]
    # We have to enforce the covariance to be psd
    log_sigma = tf.Variable(tf.ones(shape=[n_mixtures,dim], dtype=dtype))# shape: [n_mixtures,1]
    sigma = tf.matrix_diag(tf.exp(log_sigma))# shape: [n_mixtures,dim,dim]
    """
    # General psd sigma
    A = tf.Variable(tf.random_normal([n_mixtures,dim,dim],mean=0.0, stddev=0.01, dtype=dtype))# shape: [n_mixtures,dim,dim]
    sigma = 0.5 * (tf.exp(A) + tf.transpose(tf.exp(A),perm=[0,2,1])) + dim*tf.eye(dim,batch_shape=[n_mixtures])# shape: [n_mixtures,dim,dim]
    """
    return tf.concat([mu,sigma],axis=-1)# shape: [n_mixtures,dim,1+dim]

def sample_gaussian(mean_params,dtype=tf.float32):
    """
    Sample guassian given mean and covariance
    """
    mu = tf.expand_dims(mean_params[:,:,:,0],axis=-1)# shape: [batch,K,N,1]
    shape = mu.get_shape().as_list()
    sigma = mean_params[:,:,:,1:]# shape: [batch,K,N,N]
    chol = tf.cholesky(sigma)# shape: [batch,K,N,N]
    norm = tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=dtype)# shape: [batch,K,N,1]
    return tf.add(mu,tf.matmul(chol,norm))# shape: [batch,K,N,1]

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
    def __init__(self,recog_archi, gener_archi, K=10, N=20, P=784, max_iter=20):
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
        # Initialize weights of the NNs
        self._init_weights(recog_archi,gener_archi)

    def _set_distributions(self):
        """
        Define the distributions of the model and initialize the parameters
        """
        # ExpFam used in the model. Handles the natparam and expstats computations
        self.labels = distributions.discrete()
        self.gaussian = distributions.gaussian()

    def _init_weights(self,recog_archi,gener_archi):
        # Initialize nn parameters
        self.recognitionnet = nn.dense_net(recog_archi, "recog",data_type())
        self.generatornet = nn.dense_net(gener_archi, "gener",data_type())

    def _init_params(self):
        # Initialize parameters of the model \theta = (\pi, {(\mu_k, \Sigma_k)}_{k=1}^K)
        gauss_global_mean   = init_gaussian(self.K,self.N,dtype=data_type())# shape: [K,N,1+N]
        label_global_mean   = init_cat(self.K,"global",dtype=data_type())# shape: [K,1]
        init_label_stats    = init_cat(self.K,"variational",dtype=data_type())# shape: [K,1]
        return init_label_stats,label_global_mean,gauss_global_mean

    def _build_recognition_net(self,input_):
        # Foward pass for recognition network and get potential
        recog_out = self.recognitionnet._build_network(input_)# shape: [batch,2N]
        # Outout of NN encode mu and the diag of sigma
        recog_mu = tf.reshape(recog_out[:,:self.N],[-1,1,self.N,1])# shape: [batch,1,N,1]
        # we enforce it to be nsd
        recog_sigma = tf.expand_dims(tf.matrix_diag(-tf.exp(recog_out[:,self.N:])),axis=1)# shape: [batch,1,N,N]
        """
        # General psd sigma
        log_recog_sigma = devec(tf.expand_dims(recog_out[:,self.N:],axis=1),self.N,self.N)# shape: [batch,1,N,N]
        # we enforce it to be nsd
        recog_sigma = -(0.5 * (tf.exp(log_recog_sigma) + tf.transpose(tf.exp(log_recog_sigma),perm=[0,1,3,2])) + self.N*tf.eye(self.N,batch_shape=[1,1]))# shape: [batch,1,N,N]
        """
        return tf.squeeze(vec(tf.concat([recog_mu,recog_sigma],axis=-1)),axis=1)# shape: [batch,N(N+1)]

    def _build_generator_net(self,input_):
        # Foward pass for generator network
        y_reconstr_mean = self.generatornet._build_network(input_)# shape: [batch,IMAGE_SIZE*IMAGE_SIZE]
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

    def _meanfield_fixed_point(self,node_potential,gaussian_global,label_global,label_stats_):
        """
        Compute partially optimizers of the surrogate objective.
        Return the expected labels stats need for the inference
        shape:
            - gaussian_global:  [batch,K,N,N+1]
            - label_global:     [batch,K,1]
            - label_stats:     [batch,K,1]
        """
        label_stats = label_stats_
        self.kl_list = []
        self.gauss_nat_list = []
        self.label_nat_list = []
        self.gauss_stats_list = []
        self.label_stats_list = []
        # block ascent to find partial optimizers
        for i in range(self.max_iter):
            gaussian_natparam, gaussian_stats, gaussian_kl = \
                self._gaussian_meanfield(gaussian_global, node_potential, label_stats)
            label_natparam, label_stats, label_kl = \
                self._label_meanfield(label_global, gaussian_global, gaussian_stats)

            self.kl_list.append(gaussian_kl+label_kl)
            self.gauss_nat_list.append(gaussian_natparam)
            self.label_nat_list.append(label_natparam)
            self.gauss_stats_list.append(gaussian_stats)
            self.label_stats_list.append(label_stats)

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
        # get gaussian expected stats from updated natparams
        stats = self.gaussian.expectedstats(natparam)# shape: [batch,1,N,N+1]
        # compute gaussian kl
        stats_flat = vec(stats)
        dot_product = tf.reduce_sum(tf.multiply(tf.squeeze(stats_flat,axis=1),node_potential),axis=-1,keep_dims=True)# shape: [batch,1]
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
        dot_product = tf.squeeze(tf.matmul(gauss_potentials,tf.transpose(gaussian_stats_flat,perm=[0,2,1])),axis=-1)
        kl = dot_product - (self.labels.logZ(natparam)-self.labels.logZ(label_global))# shape: [batch,1]
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
        self.pot = node_potential
        # Compute partial optimizers for surrogate objective
        label_stats = self._meanfield_fixed_point(node_potential,gaussian_global,label_global,label_stats_init)# shape: [batch,K,1]
        # Compute local KL with partial otimized parameters
        local_KL, (labels_natparams,gaussian_natparams) = self._local_meanfield(node_potential,gaussian_global,label_global,label_stats)
        self.gauss_test = self.gaussian.natural_to_standard(gaussian_natparams)
        # Sample x from q, pass to generatornet
        x = sample_gaussian(self.gaussian.natural_to_standard(gaussian_natparams),data_type())# shape: [batch,1,N,1]
        # Build generator network and compute params of the obs variables from samples
        logits = self._build_generator_net(tf.squeeze(x))# shape: [batch,IMAGE_SIZE*IMAGE_SIZE]
        self.y_reconstr_mean = tf.nn.sigmoid(logits)
        # Compute loglikeihood term
        loglikelihood = -tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
        # Compute SVAE objective
        self.SVAE_obj = tf.reduce_sum(loglikelihood - local_KL)# sum over batch
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-self.SVAE_obj,global_step=batch)

    def _generate(self,gaussian_mean):
        """
        Generate sample from trained gaussian_mean
        shape:
            - gaussian_mean:  [nsamples,K,N,N+1]
        """
        """
        x = tf.squeeze(sample_gaussian(gaussian_mean,data_type()),axis=-1)# shape: [1,K,N]
        x_list = tf.unstack(x,num=None,axis=1)# shape: K*[1,N]
        # Build generator network and compute params of the obs variables from samples
        logits = tf.stack([tf.squeeze(self._build_generator_net(x_list[i])) for i in range(len(x_list))],axis=1)# shape: [K,IMAGE_SIZE*IMAGE_SIZE]
        self.y_generate_mean = tf.sigmoid(logits)# shape: [1,K,IMAGE_SIZE*IMAGE_SIZE]
        """
        x = tf.squeeze(sample_gaussian(gaussian_mean,data_type()))# shape: [K,N]
        # Build generator network and compute params of the obs variables from samples
        logits = self._build_generator_net(x)# shape: [K,IMAGE_SIZE*IMAGE_SIZE]
        self.y_generate_mean = tf.sigmoid(logits)# shape: [K,IMAGE_SIZE*IMAGE_SIZE]
