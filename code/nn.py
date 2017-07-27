import numpy as np
import tensorflow as tf
import abc

import pdb

# Xavier initialization of weights
def xavier_init(fan_in, fan_out, constant=1, dtype=tf.float32):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                                    minval=low,
                                    maxval=high,
                                    dtype=dtype)

class nnet(metaclass=abc.ABCMeta):
    """
    Abstract class for neural network
    Args:
    input: input of the nn_net
    archi: architecture of the nnet
    """
    def __init__(self, nnarchi, name, dtype=tf.float32):
        self.archi = nnarchi
        self.name = name
        self.dtype = dtype

    @abc.abstractmethod
    def _build_network(self):
        """
        Main function, build the nnet
        """
        pass

    @abc.abstractmethod
    def _initialize_weights(self):
        """
        Initialize the weights of the nnet
        """
        pass

    @abc.abstractmethod
    def _build_graph(self):
        """
        Build graph of the nnet
        """
        pass

class dense_net(nnet):
    def __init__(self, nnarchi, name, dtype=tf.float32):
        super().__init__(nnarchi, name, dtype=tf.float32)

    def _build_network(self, nninput):
        weights = self._initialize_weights()
        nn_output = self._build_graph(nninput,weights["weights_"+self.name],weights["biases_"+self.name])
        return nn_output

    def _initialize_weights(self):
        all_weights = dict()
        all_weights["weights_" + self.name] = {
            'h1': tf.Variable(xavier_init(self.archi["ninput"], self.archi["nhidden_1"])),
            'h2': tf.Variable(xavier_init(self.archi["nhidden_1"], self.archi["nhidden_2"])),
            'out': tf.Variable(xavier_init(self.archi["nhidden_2"], self.archi["noutput"]))}
        all_weights["biases_" + self.name] = {
            'b1': tf.Variable(tf.zeros([self.archi["nhidden_1"]], dtype=self.dtype)),
            'b2': tf.Variable(tf.zeros([self.archi["nhidden_2"]], dtype=self.dtype)),
            'out': tf.Variable(tf.zeros([self.archi["noutput"]], dtype=self.dtype))}
        return all_weights

    def _build_graph(self, nninput, weights, biases):
        layer_1 = tf.nn.elu(tf.add(tf.matmul(nninput,
                                            weights['h1']),
                                            biases['b1']))
        layer_2 = tf.nn.elu(tf.add(tf.matmul(layer_1,
                                            weights['h2']),
                                            biases['b2']))
        nn_output = tf.add(tf.matmul(layer_2, weights['out']),
                                            biases['out'])
        return nn_output
