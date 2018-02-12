import numpy as np
from network import activations


class Layer:

    def __init__(self, number, activation=None, dropout=0.0):
        self.number = number
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

    def __dropout__(self, a):
        if self.dropout == 0:
            return a
        return a * np.random.binomial(np.ones_like(a, int), 1 - self.dropout) * (1 / (1 - self.dropout))


class InputLayer(Layer):

    def __init__(self, number, dropout=0.0):
        Layer.__init__(self, number, dropout=dropout)

    def feedforward(self, a):
        return a

    def activate(self, a):
        return self.__dropout__(a)

    def bind(self, layer):
        pass


class HiddenLayer(Layer):

    def __init__(self, number, activation=activations.Sigmoid, dropout=0.0):
        Layer.__init__(self, number, activation, dropout)

    def clone(self, number):
        clones = []

        for i in range(number):
            clones.append(HiddenLayer(self.number, self.activations))

        return clones


class OutputLayer(Layer):

    def __init__(self, number, activation=activations.Sigmoid):
        Layer.__init__(self, number, activation)
