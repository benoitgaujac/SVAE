import numpy as np
import tensorflow as tf
import abc
from math import ceil

import pdb

# Normal initialization of weights
def weight_variable(shape,name):
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

class dense_net_3l(nnet):
    def __init__(self, nnarchi, name, dtype=tf.float32):
        super().__init__(nnarchi, name, dtype=tf.float32)

    def _initialize_weights(self):
        self.all_weights = dict()
        self.all_weights["weights_" + self.name] = {
            'h1': weight_variable([self.archi["ninput"], self.archi["nhidden_1"]],self.name + "_weights0"),
            'h2': weight_variable([self.archi["nhidden_1"], self.archi["nhidden_2"]],self.name + "_weights1"),
            'h3': weight_variable([self.archi["nhidden_2"], self.archi["nhidden_3"]],self.name + "_weights2"),
            'out': weight_variable([self.archi["nhidden_3"], self.archi["noutput"]],self.name + "_weights3")}
        self.all_weights["biases_" + self.name] = {
            'b1': bias_variable([self.archi["nhidden_1"]], self.name + "_biais0"),
            'b2': bias_variable([self.archi["nhidden_2"]], self.name + "_biais1"),
            'b3': bias_variable([self.archi["nhidden_3"]], self.name + "_biais2"),
            'out': bias_variable([self.archi["noutput"]], self.name + "_biais3")}

    def _build_network(self, nninput):
        nn_output = self._build_graph(nninput,self.all_weights["weights_"+self.name],self.all_weights["biases_"+self.name])
        return nn_output

    def _build_graph(self, nninput, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(nninput,
                                            weights['h1']),
                                            biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,
                                            weights['h2']),
                                            biases['b2']))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2,
                                            weights['h3']),
                                            biases['b3']))
        nn_output = tf.add(tf.matmul(layer_3, weights['out']),
                                            biases['out'])
        return nn_output

class conv_net(nnet):
    def __init__(self, nnarchi, name, dtype=tf.float32):
        super().__init__(nnarchi, name, dtype=tf.float32)

    def _initialize_weights(self):
        self.all_weights = dict()
        self.all_weights["weights_" + self.name] = {
            'h1': weight_variable([self.archi["size_filters"],self.archi["size_filters"],1,self.archi["num_filters"]], self.name + "_weights0"),
            'h2': weight_variable([self.archi["size_filters"],self.archi["size_filters"],self.archi["num_filters"], self.archi["num_filters"]],self.name + "_weights1"),
            'h3': weight_variable([self.archi["num_filters"]*self.archi["ninput"]*self.archi["ninput"],self.archi["fc"]], self.name + "_weights2"),
            'out': weight_variable([self.archi["fc"], self.archi["noutput"]], self.name + "_weights3")}
        self.all_weights["biases_" + self.name] = {
            'b1': bias_variable([self.archi["num_filters"]], self.name + "_biais0"),
            'b2': bias_variable([self.archi["num_filters"]], self.name + "_biais1"),
            'b3': bias_variable([self.archi["fc"]], self.name + "_biais2"),
            'out': bias_variable([self.archi["noutput"]], self.name + "_biais3")}

    def _build_network(self, nninput):
        nn_output = self._build_graph(nninput,self.all_weights["weights_"+self.name],self.all_weights["biases_"+self.name])
        return nn_output

    def _build_graph(self, nninput, weights, biases):
        # Conv 1
        conv_1 = tf.nn.conv2d(nninput,weights["h1"],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        layer_1 = tf.nn.relu(tf.add(conv_1,biases['b1']))
        # Conv 2
        conv_2 = tf.nn.conv2d(layer_1,weights["h2"],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        layer_2 = tf.nn.relu(tf.add(conv_1,biases['b2']))
        # Flatten
        layer_2_flatten = tf.reshape(layer_2,[-1,self.archi["num_filters"]*self.archi["ninput"]*self.archi["ninput"]])
        # FC 1
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_flatten,weights['h3']),biases['b3']))
        # output
        nn_output = tf.add(tf.matmul(layer_3, weights['out']),biases['out'])
        return nn_output

class conv_net_4l(nnet):
    def __init__(self, nnarchi, name, dtype=tf.float32):
        super().__init__(nnarchi, name, dtype=tf.float32)

    def _initialize_weights(self):
        self.all_weights = dict()
        self.all_weights["weights_" + self.name] = {
            'h1': weight_variable([self.archi["size_filters_1"],self.archi["size_filters_1"],1,self.archi["num_filters"]], self.name + "_weights0"),
            'h2': weight_variable([self.archi["size_filters_2"],self.archi["size_filters_2"],self.archi["num_filters"],2*self.archi["num_filters"]], self.name + "_weights1"),
            'h3': weight_variable([self.archi["size_filters_2"],self.archi["size_filters_2"],2*self.archi["num_filters"],4*self.archi["num_filters"]], self.name + "_weights2"),
            'h4': weight_variable([4*self.archi["num_filters"]*ceil(self.archi["ninput"]/8)*ceil(self.archi["ninput"]/8),self.archi["fc"]], self.name + "_weights3"),
            'out': weight_variable([self.archi["fc"], self.archi["noutput"]], self.name + "_weights4")}
        self.all_weights["biases_" + self.name] = {
            'b1': bias_variable([self.archi["num_filters"]], self.name + "_biais0"),
            'b2': bias_variable([2*self.archi["num_filters"]], self.name + "_biais1"),
            'b3': bias_variable([4*self.archi["num_filters"]], self.name + "_biais2"),
            'b4': bias_variable([self.archi["fc"]], self.name + "_biais3"),
            'out': bias_variable([self.archi["noutput"]], self.name + "_biais4")}

    def _build_network(self, nninput):
        nn_output = self._build_graph(nninput,self.all_weights["weights_"+self.name],self.all_weights["biases_"+self.name])
        return nn_output

    def _build_graph(self, nninput, weights, biases):
        # Conv 1
        conv_1 = tf.nn.conv2d(nninput,weights["h1"],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        layer_1 = tf.nn.relu(tf.add(conv_1,biases['b1']))
        # Conv 2
        conv_2 = tf.nn.conv2d(layer_1,weights["h2"],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        layer_2 = tf.nn.relu(tf.add(conv_2,biases['b2']))
        # Conv 3
        conv_3 = tf.nn.conv2d(layer_2,weights["h3"],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        layer_3 = tf.nn.relu(tf.add(conv_3,biases['b3']))
        # Flatten
        layer_3_flatten = tf.reshape(layer_3,[-1,4*self.archi["num_filters"]*ceil(self.archi["ninput"]/8)*ceil(self.archi["ninput"]/8)])
        # FC 1
        layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3_flatten,weights['h4']),biases['b4']))
        # output
        nn_output = tf.add(tf.matmul(layer_4, weights['out']),biases['out'])
        return nn_output


class deconv_net(nnet):
    def __init__(self, nnarchi, name, dtype=tf.float32):
        super().__init__(nnarchi, name, dtype=tf.float32)

    def _initialize_weights(self):
        self.all_weights = dict()
        self.all_weights["weights_" + self.name] = {
            'h1': weight_variable([self.archi["ninput"], self.archi["fc"]], self.name + "_weights0"),
            'h2': weight_variable([self.archi["fc"],self.archi["num_filters"]*self.archi["noutput"]*self.archi["noutput"]], self.name + "_weights1"),
            'h3': weight_variable([self.archi["size_filters"],self.archi["size_filters"],self.archi["num_filters"],self.archi["num_filters"]], self.name + "_weights2"),
            'out': weight_variable([self.archi["size_filters"],self.archi["size_filters"],1,self.archi["num_filters"]],self.name + "_weights3")}
        self.all_weights["biases_" + self.name] = {
            'b1': bias_variable([self.archi["fc"]], self.name + "_biais0"),
            'b2': bias_variable([self.archi["num_filters"]*self.archi["noutput"]*self.archi["noutput"]], self.name + "_biais1"),
            'b3': bias_variable([self.archi["num_filters"]], self.name + "_biais2"),
            'out': bias_variable([1], self.name + "_biais3")}

    def _build_network(self, nninput):
        nn_output = self._build_graph(nninput,self.all_weights["weights_"+self.name],self.all_weights["biases_"+self.name])
        return nn_output

    def _build_graph(self, nninput, weights, biases):
        BATCH_SIZE = tf.shape(nninput)[0]
        # Input
        layer_1 = tf.nn.relu(tf.add(tf.matmul(nninput,weights['h1']),biases['b1']))
        # FC 1
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
        # Reshape FC 1
        layer_2_reshape = tf.reshape(layer_2,[BATCH_SIZE,self.archi["noutput"],self.archi["noutput"],self.archi["num_filters"]])
        # Deconv 1
        deconv_1 = tf.nn.conv2d_transpose(layer_2_reshape,
                                    weights['h3'],
                                    [BATCH_SIZE,self.archi["noutput"],self.archi["noutput"],self.archi["num_filters"]],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        layer_3 = tf.nn.relu(tf.add(deconv_1,biases['b3']))
        # Deconv 2
        deconv_2 = tf.nn.conv2d_transpose(layer_3,
                                    weights['out'],
                                    [BATCH_SIZE,self.archi["noutput"],self.archi["noutput"],1],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
        nn_output = tf.add(deconv_2,biases['out'])
        return nn_output

class deconv_net_4l(nnet):
    def __init__(self, nnarchi, name, dtype=tf.float32):
        super().__init__(nnarchi, name, dtype=tf.float32)

    def _initialize_weights(self):
        self.all_weights = dict()
        self.all_weights["weights_" + self.name] = {
            'h1': weight_variable([self.archi["ninput"], self.archi["fc"]], self.name + "_weights0"),
            'h2': weight_variable([self.archi["fc"],4*self.archi["num_filters"]*ceil(self.archi["noutput"]/8)*ceil(self.archi["noutput"]/8)], self.name + "_weights1"),
            'h3': weight_variable([self.archi["size_filters_1"],self.archi["size_filters_1"],2*self.archi["num_filters"],4*self.archi["num_filters"]], self.name + "_weights2"),
            'h4': weight_variable([self.archi["size_filters_2"],self.archi["size_filters_2"],self.archi["num_filters"],2*self.archi["num_filters"]], self.name + "_weights3"),
            'out': weight_variable([self.archi["size_filters_2"],self.archi["size_filters_2"],1,self.archi["num_filters"]],self.name + "_weights4")}
        self.all_weights["biases_" + self.name] = {
            'b1': bias_variable([self.archi["fc"]], self.name + "_biais0"),
            'b2': bias_variable([4*self.archi["num_filters"]*ceil(self.archi["noutput"]/8)*ceil(self.archi["noutput"]/8)], self.name + "_biais1"),
            'b3': bias_variable([2*self.archi["num_filters"]], self.name + "_biais2"),
            'b4': bias_variable([self.archi["num_filters"]], self.name + "_biais3"),
            'out': bias_variable([1], self.name + "_biais4")}

    def _build_network(self, nninput):
        nn_output = self._build_graph(nninput,self.all_weights["weights_"+self.name],self.all_weights["biases_"+self.name])
        return nn_output

    def _build_graph(self, nninput, weights, biases):
        BATCH_SIZE = tf.shape(nninput)[0]
        # Input
        layer_1 = tf.nn.relu(tf.add(tf.matmul(nninput,weights['h1']),biases['b1']))
        # FC 1
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
        # Reshape FC 1
        layer_2_reshape = tf.reshape(layer_2,[BATCH_SIZE,ceil(self.archi["noutput"]/8),ceil(self.archi["noutput"]/8),4*self.archi["num_filters"]])
        # Deconv 1
        deconv_1 = tf.nn.conv2d_transpose(layer_2_reshape,
                                    weights['h3'],
                                    [BATCH_SIZE,int(self.archi["noutput"]/4),int(self.archi["noutput"]/4),2*self.archi["num_filters"]],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        layer_3 = tf.nn.relu(tf.add(deconv_1,biases['b3']))
        # Deconv 2
        deconv_2 = tf.nn.conv2d_transpose(layer_3,
                                    weights['h4'],
                                    [BATCH_SIZE,int(self.archi["noutput"]/2),int(self.archi["noutput"]/2),self.archi["num_filters"]],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        layer_3 = tf.nn.relu(tf.add(deconv_2,biases['b4']))
        # Deconv 3
        deconv_3 = tf.nn.conv2d_transpose(layer_3,
                                    weights['out'],
                                    [BATCH_SIZE,self.archi["noutput"],self.archi["noutput"],1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        nn_output = tf.add(deconv_3,biases['out'])
        return nn_output
