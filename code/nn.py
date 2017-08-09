import numpy as np
import tensorflow as tf
import abc

import pdb

# Normal initialization of weights
def weight_variable(shape,name):
    if name[0]=="r":
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
    elif name[0]=="g":
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
    return tf.get_variable(name, shape, initializer=initializer)

def bias_variable(shape,name):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name, shape, initializer=initializer)


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
        self._initialize_weights()

    @abc.abstractmethod
    def _initialize_weights(self):
        """
        Initialize the weights of the nnet
        """
        pass

    @abc.abstractmethod
    def _build_network(self):
        """
        Main function, build the nnet
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

    def _initialize_weights(self):
        self.all_weights = dict()
        self.all_weights["weights_" + self.name] = {
            'h1': weight_variable([self.archi["ninput"], self.archi["nhidden_1"]],self.name + "_weights0"),
            'h2': weight_variable([self.archi["nhidden_1"], self.archi["nhidden_2"]],self.name + "_weights1"),
            'out': weight_variable([self.archi["nhidden_2"], self.archi["noutput"]],self.name + "_weights2")}
        self.all_weights["biases_" + self.name] = {
            'b1': bias_variable([self.archi["nhidden_1"]], self.name + "_biais0"),
            'b2': bias_variable([self.archi["nhidden_2"]], self.name + "_biais1"),
            'out': bias_variable([self.archi["noutput"]], self.name + "_biais2")}

    def _build_network(self, nninput):
        nn_output = self._build_graph(nninput,self.all_weights["weights_"+self.name],self.all_weights["biases_"+self.name])
        return nn_output

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
