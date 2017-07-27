import numpy as np
import tensorflow as tf

import pdb

import distributions
import nn



## Initialize seeds
np.random.seed(0)
tf.set_random_seed(0)
eps = 0.0000001


######################################## Utils functions ########################################
# Initialization of discrete params
def init_cat(n_mixtures,dtype=tf.float32):
    """
    Initialize parameters of discrete distribution following dirichlet
    """
    discr_mean = tf.Variable(tf.ones(shape=[n_mixtures,1], dtype=dtype))# shape: [n_mixtures,1]
    # we use softmax to ensure the mean components sum to one
    soft_max = tf.nn.softmax(discr_mean)
    return soft_max
    """
    mean_list = [soft_max for i in range(batch_size)]
    return tf.stack(mean_list,axis=0)# shape: [batch,n_mixtures,1]
    """

# Initialization of mean and covariance matrix
def init_gaussian(n_mixtures,dim,dtype=tf.float32):
    """
    Initialize means and covariance matrix for the gaussian mixtures components
    """
    mu = tf.Variable(tf.random_normal([n_mixtures,dim,1], mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,dim,1]
    # We have to enforce the covariance to be psd
    diag = tf.Variable(tf.random_normal([n_mixtures,dim],mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,dim,dim]
    sigma = tf.matrix_diag(tf.square(diag))# shape: [n_mixtures,dim,dim]
    sigma = tf.maximum(eps*tf.eye(dim,batch_shape=[n_mixtures],dtype=dtype),sigma)
    """
    If we want sigma to be not diag only
    #sigma = tf.Variable(tf.random_normal([n_mixtures,dim,dim],mean=0.0, stddev=1.0, dtype=dtype))# shape: [n_mixtures,dim,dim]
    #sigma = tf.matmul(sigma, tf.transpose(sigma,perm=[0,2,1]))
    #sigma = tf.add(sigma,tf.scalar_mul(eps,tf.eye(dim,dtype=dtype)))# shape: [n_mixtures,dim,dim]
    """
    return tf.concat([mu,sigma],axis=-1) # shape: [n_mixtures,dim,1+dim]

def sample_gaussian(mean_params,dtype=tf.float32):
    """
    Sample guassian given mean and covariance
    """
    mu = tf.squeeze(mean_params[:,:,:,0])
    shape = mu.get_shape().as_list() + [1]
    sigma = mean_params[:,:,:,1:]
    norm = tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=dtype)
    chol = tf.cholesky(tf.squeeze(sigma))
    return tf.add(mu,tf.squeeze(tf.matmul(chol,norm),axis=-1))

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
    def __init__(self, K=10, N=20, P=784,learning_rate=0.001,
                                            batch_size=100):
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Define the distributions of the model
        self._set_distributions()

    def _set_distributions(self):
        """
        Define the distributions of the model and initialize the parameters
        """
        # ExpFam used in the model. Handles the natparam and expstats computations
        self.labels = distributions.discrete()
        self.gaussian = distributions.gaussian()

    def _initit_params(self,recog_archi,gener_archi):
        # Initialize nn parameters
        self.recognitionnet = nn.dense_net(recog_archi, "recog",data_type())
        self.generatornet = nn.dense_net(gener_archi, "gener",data_type())
        # Initialize parameters of the model \theta = (\pi, {(\mu_k, \Sigma_k)}_{k=1}^K)
        gauss_mean = init_gaussian(self.K,self.N,dtype=data_type())# shape: [n_mixtures,dim,1+dim]
        cat_mean = init_cat(self.K,dtype=data_type())# shape: [n_mixtures,1]
        return cat_mean,gauss_mean

    def _build_recognition_net(self,input_):
        # Build recognition network and get potential
        recog_out = self.recognitionnet._build_network(input_)# shape: [batch,2N]
        # Outout of NN encode mu and the diag of sigma
        recog_mu = tf.reshape(recog_out[:,:self.N],[-1,1,self.N,1])# shape: [batch,1,N,1]
        recog_sigma = tf.expand_dims(tf.matrix_diag(tf.square(recog_out[:,self.N:])),axis=1)# shape: [batch,1,N,N]
        node_potential = self.gaussian.standard_to_natural(tf.concat([recog_mu,recog_sigma],axis=-1))# shape: [batch,1,N,1+N]
        node_potential = tf.reshape(node_potential,[-1,self.N*(self.N+1)])# shape: [batch,N*(1+N)]
        return node_potential

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
        max_iter=20
        label_stats_ = label_stats
        # block ascent to find partial optimizers
        for i in range(max_iter):
            gaussian_natparam, gaussian_stats, gaussian_kl = \
                self._gaussian_meanfield(gaussian_global, node_potential, label_stats_)
            label_natparam, label_stats_, label_kl = \
                self._label_meanfield(label_global, gaussian_global, gaussian_stats)
        return label_stats_

    def _gaussian_meanfield(self,gaussian_global, node_potential, label_stats):
        """
        Compute gaussian variational factor
        shape:
            - gaussian_global:  [batch,K,N,N+1]
            - node_potential:   [batch,N*(N+1)]
            - label_stats:      [batch,K,1]
        """
        # compute the potential
        gaussian_global_flat = tf.reshape(gaussian_global,[-1,self.K,self.N*(self.N+1)])# shape: [batch,K,N(N+1)]
        global_potentials = tf.squeeze(tf.matmul(tf.transpose(label_stats,perm=[0,2,1]),gaussian_global_flat),axis=1)# shape: [batch,N(N+1)]
        # update gaussian natparams
        natparam = tf.reshape(tf.add(node_potential,global_potentials),[-1,1,self.N,self.N+1])# shape: [batch,1,N,N+1]
        # get gaussian expected stats from updated natparams
        stats = self.gaussian.expectedstats(natparam)# shape: [batch,1,N,N+1]
        # compute gaussian kl
        stats_flat = tf.reshape(stats, [-1,self.N*(self.N+1)])# shape: [batch,N*(N+1)]
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
        gaussian_global_flat = tf.reshape(gaussian_global,[-1,self.K,self.N*(self.N+1)])# shape: [batch,K,N(N+1)]
        gaussian_stats_flat = tf.reshape(gaussian_stats, [-1,1,self.N*(self.N+1)])# shape: [batch,1,N(N+1)]
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

    def _create_loss_optimizer(self,gaussian_global,label_global,label_stats_init,y):
        """
        Compute the SVAE objective using reparametrization trick for likelyhood term
        shape:
            - gaussian_global:  [batch,K,N,N+1]
            - label_global:     [batch,K,1]
            - label_stats_init:     [batch,K,1]
        """
        # Build recognition network and compute node_potential
        self.node_potential = self._build_recognition_net(y)# shape: [batch,N(N+1)]
        # Compute partial optimizers for surrogate objective
        label_stats = self._meanfield_fixed_point(self.node_potential,gaussian_global,label_global,label_stats_init)# shape: [batch,K,1]
        # Compute local KL with partial otimized parameters
        local_KL, (labels_natparams,gaussian_natparams) = self._local_meanfield(self.node_potential,gaussian_global,label_global,label_stats)
        # Sample x from q, pass to generatornet
        self.gauss_mean = self.gaussian.natural_to_standard(gaussian_natparams)
        x = sample_gaussian(self.gauss_mean)# shape: [batch,N]
        # Build generator network and compute params of the obs variables from samples
        logits = self._build_generator_net(x)# shape: [batch,IMAGE_SIZE*IMAGE_SIZE]
        self.y_reconstr_mean = tf.sigmoid(logits)
        # Compute loglikeihood term
        loglikelihood = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
        # Compute SVAE objective
        self.SVAE_obj = tf.reduce_mean(loglikelihood) - tf.reduce_sum(local_KL)# average over batch
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.SVAE_obj)
