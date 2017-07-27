import tensorflow as tf
import abc

import pdb

pi = 3.14159265359

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
        Compute natparams from mean params (mu,sigma)
        shape:
            - mu = mean_params[0]:      [batch,K,N,1]
            - sigma = mean_params[1]:   [batch,K,N,N]
        """
        mu = mean_params[0]# shape: [batch,n_mixtures,dim,1]
        sigma = mean_params[1] # shape: [batch,n_mixtures,dim,dim]
        inverse_sigma = tf.matrix_inverse(sigma) # shape: [batch,n_mixtures,dim,dim]
        sigmu = tf.matmul(inverse_sigma, mu) # shape: [batch,n_mixtures,dim,1]
        gaussian_natparams = tf.concat([sigmu,tf.scalar_mul(-0.5,inverse_sigma)],axis=3) # shape: [batch,n_mixtures,dim,1+dim]
        return gaussian_natparams # shape: [batch,n_mixtures,dim,1+dim]

    def natural_to_standard(self, nat_params):
        """
        Compute mean params from natparams
        shape:
            - nat_params:      [batch,K,N,N+1]
        """
        sigmu = tf.expand_dims(nat_params[:,:,:,0],axis=-1) # shape: [batch,n_mixtures,dim,1]
        sig_inverse = tf.scalar_mul(-2,nat_params[:,:,:,1:]) # shape: [batch,n_mixtures,dim,dim]
        sigma = tf.matrix_inverse(sig_inverse) # shape: [batch,n_mixtures,dim,dim]
        mu = tf.matmul(sigma,sigmu)# shape: [batch,n_mixtures,dim,1]
        return (mu,sigma)

    def expectedstats(self, nat_params):
        """
        Compute expected sufficient stats from natparams
        shape:
            - nat_params:      [batch,K,N,N+1]
        """
        (mu,sigma) = self.natural_to_standard(nat_params) # shape: [batch,n_mixtures,dim,1], [batch,n_mixtures,dim,dim]
        mumut = tf.matmul(mu,tf.transpose(mu, perm=[0,1,3,2])) # shape: [batch,n_mixtures,dim,dim]
        return tf.concat([mu,tf.add(sigma,mumut)],axis=3) # shape: [batch,n_mixtures,dim,1+dim]

    def logZ(self, nat_params):
        """
        Compute log partition function from natparams
        shape:
            - nat_params:      [batch,K,N,N+1]
        """
        (mu,sigma) = self.natural_to_standard(nat_params) # shape: [batch,n_mixtures,dim,1], [batch,n_mixtures,dim,dim]
        #logdet(2*pi*sigma))
        log_det = tf.expand_dims(tf.log(tf.matrix_determinant(tf.scalar_mul(2*pi,sigma))),axis=-1) # shape: [batch,n_mixtures,1]
        # Quadratic term
        mu_t = tf.transpose(mu, perm=[0,1,3,2]) # shape: [batch,n_mixtures,1,dim]
        sig_inverse = tf.matrix_inverse(sigma) # shape: [batch,n_mixtures,dim,dim]
        musigmu = tf.squeeze(tf.matmul(tf.matmul(mu_t,sig_inverse),mu),axis=-1) # shape: [batch,n_mixtures,1]
        return tf.scalar_mul(0.5,tf.add(musigmu,log_det))

class discrete(distributions):

    def standard_to_natural(self, standard_params):
        mean = standard_params
        label_natparams = tf.log(mean)
        return label_natparams

    def natural_to_standard(self, nat_params):
        logmean = nat_params
        label_standard = tf.exp(logmean)
        return label_standard

    def expectedstats(self, nat_params):
        logmean = nat_params
        label_expectedstats = tf.exp(logmean)
        return label_expectedstats

    def logZ(self,nat_params):
        return 0.
