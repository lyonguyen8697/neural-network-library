import numpy as np
from network import activations


class _Layer:

    def __init__(self, numbers, activation=None, dropout=0.0):
        self.numbers = numbers
        self.activation = activation() if activation is not None else None
        self.dropout = dropout

        self.weights = []
        self.biases = []

        self.next = None
        self.previous = None

    def feedforward(self, a):
        return self.activation.evaluate(np.dot(self.weights, a) + self.biases)

    def activate(self, a):
        z = np.dot(self.weights, a) + self.biases
        a = self.activation.evaluate(z)
        return self.__dropout__(a), z

    def bind(self, layer):
        layer.next = self
        self.previous = layer

    def serialize(self):
        return {
            'numbers': self.numbers,
            'activation': self.activation.serialize(),
            'dropout': self.dropout,
            'weights': self.weights.tolist(),
            'biases': self.biases.tolist(),
        }

    def __dropout__(self, a):
        if self.dropout == 0:
            return a
        return a * np.random.binomial(np.ones_like(a, int), 1 - self.dropout) * (1 / (1 - self.dropout))


class InputLayer(_Layer):

    def __init__(self, numbers, dropout=0.0):
        _Layer.__init__(self, numbers, dropout=dropout)

    def feedforward(self, a):
        return a

    def activate(self, a):
        return self.__dropout__(a)

    def bind(self, layer):
        pass

    def serialize(self):
        data = _Layer.serialize()
        data.pop('activation')
        data.pop('weights')
        data.pop('biases')
        return data


class HiddenLayer(_Layer):

    def __init__(self, numbers, activation=activations.Sigmoid, dropout=0.0):
        _Layer.__init__(self, numbers, activation, dropout)

    def clone(self, number):
        clones = []

        for i in range(number):
            clones.append(HiddenLayer(self.number, self.activation))

        return clones


# class ConvolutionalLayer:
#
#     def __init__(self, feature_maps, local_receptive_field, stride_length):
#         self.feature_maps = feature_maps
#         self.local_receptive_field = local_receptive_field
#         self.stride_length = stride_length
#
#     def feedforward(self, a):
#         np.reshape()
#         np.sum()
#
#
# class PoolingLayer:
#
#     def __init__(self):
#         pass


class OutputLayer(_Layer):

    def __init__(self, numbers, activation=activations.Sigmoid):
        _Layer.__init__(self, numbers, activation)

    def serialize(self):
        data = _Layer.serialize()
        data.pop('dropout')
        return data
