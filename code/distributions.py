import tensorflow as tf
import abc

import pdb
import svae

pi = 3.14159265359
eps = svae.eps

class distributions(metaclass=abc.ABCMeta):
    """
    Abstract class for distribution
    """

    @abc.abstractmethod
    def standard_to_natural(self,standard_params):
        """
        Convert standard params to natural params form for exponential fam
        """
        pass

    @abc.abstractmethod
    def natural_to_standard(self,nat_params):
        """
        Convert natural params to standard params form for exponential fam
        """
        pass

    @abc.abstractmethod
    def expectedstats(self, nat_params):
        """
        Compute the expected sufficient stats
        """
        pass

    @abc.abstractmethod
    def logZ(self, nat_params):
        """
        Compute the log-partition function
        """
        pass

class gaussian(distributions):
    def standard_to_natural(self,mean_params):
        """
        Compute natparams from mean params
        shape:
            - mean_params:      [batch,K,N,1+N]
        """
        mu = tf.expand_dims(mean_params[:,:,:,0],axis=-1)# shape: [batch,n_mixtures,dim,1]
        [K,N] = mu.get_shape().as_list()[1:3]
        inverse_sigma = tf.matrix_inverse(mean_params[:,:,:,1:]+eps*tf.eye(N,batch_shape=[K])) # shape: [batch,n_mixtures,dim,dim]
        sigmu = tf.matmul(inverse_sigma, mu) # shape: [batch,n_mixtures,dim,1]
        gaussian_natparams = tf.concat([sigmu,-0.5*inverse_sigma],axis=-1) # shape: [batch,n_mixtures,dim,1+dim]
        return gaussian_natparams # shape: [batch,n_mixtures,dim,1+dim]

    def natural_to_standard(self, nat_params):
        """
        Compute mean params from natparams
        shape:
            - nat_params:      [batch,K,N,N+1]
        """
        [K,N] = nat_params.get_shape().as_list()[1:3]
        sigma = -0.5*tf.matrix_inverse(nat_params[:,:,:,1:]+eps*tf.eye(N,batch_shape=[K])) # shape: [batch,n_mixtures,dim,dim]
        mu = tf.matmul(sigma,tf.expand_dims(nat_params[:,:,:,0],axis=-1))# shape: [batch,n_mixtures,dim,1]
        return tf.concat([mu,sigma],axis=-1) # shape: [batch,n_mixtures,dim,1+dim]

    def expectedstats(self, nat_params):
        """
        Compute expected sufficient stats from natparams
        shape:
            - nat_params:      [batch,K,N,N+1]
        """
        mean_params = self.natural_to_standard(nat_params) # shape: batch,n_mixtures,dim,1+dim]
        mu = tf.expand_dims(mean_params[:,:,:,0],axis=-1)# shape: [batch,n_mixtures,dim,1]
        sigma = mean_params[:,:,:,1:]# shape: [batch,n_mixtures,dim,dim]
        mumut = tf.matmul(mu,tf.transpose(mu, perm=[0,1,3,2])) # shape: [batch,n_mixtures,dim,dim]
        return tf.concat([mu,tf.add(sigma,mumut)],axis=-1) # shape: [batch,n_mixtures,dim,1+dim]

    def logZ(self, nat_params):
        """
        Compute log partition function from natparams
        shape:
            - nat_params:      [batch,K,N,N+1]
        """
        mean_params = self.natural_to_standard(nat_params) # shape: batch,n_mixtures,dim,1+dim]
        mu = tf.expand_dims(mean_params[:,:,:,0],axis=-1)# shape: [batch,n_mixtures,dim,1]
        sigma = mean_params[:,:,:,1:]# shape: [batch,n_mixtures,dim,dim]
        #logdet
        [K,N] = nat_params.get_shape().as_list()[1:3]
        logdet = tf.expand_dims(tf.log(tf.matrix_determinant(sigma)+eps*tf.ones([K])),axis=-1) + N*tf.log([2.0*pi])# shape: [batch,n_mixtures,1]
        # Quadratic term
        mu_t = tf.transpose(mu, perm=[0,1,3,2]) # shape: [batch,n_mixtures,1,dim]
        musigmu = tf.squeeze(tf.matmul(tf.matmul(mu_t,-2*nat_params[:,:,:,1:]),mu),axis=-1) # shape: [batch,n_mixtures,1]
        return tf.scalar_mul(0.5,tf.add(musigmu,logdet))

class discrete(distributions):

    def standard_to_natural(self, standard_params):
        mean = standard_params
        label_natparams = tf.log(mean)
        return label_natparams

    def natural_to_standard(self, nat_params):
        #logmean = nat_params
        label_standard = tf.exp(nat_params)
        return label_standard

    def expectedstats(self, nat_params):
        logmean = nat_params
        label_expectedstats = tf.exp(logmean)
        return label_expectedstats

    def logZ(self,nat_params):
        return 0.
