import numpy as np
from network import activations
import time

class InputLayer:

    def __init__(self, number):
        self.number = number

        self.next = None

    def feedforward(self, a):
        return a


class HiddenLayer:

    def __init__(self, number, activation=activations.Sigmoid):
        self.number = number
        self.activation = activation()

        self.weights = []
        self.biases = []

        self.next = None
        self.previous = None

    def feedforward(self, a):
        # return self.activation.evaluate(np.dot(self.weights, a) + self.biases)
        z = np.dot(self.weights, a) + self.biases
        start = time.clock()
        a = self.activation.evaluate(z)
        end = time.clock()
        # print(self.activation, (end - start) * 1000)
        return a

    def bind(self, layer):
        layer.next = self
        self.previous = layer

    def clone(self, number):
        clones = []

        for i in range(number):
            clones.append(HiddenLayer(self.number, self.activations))

        return clones


class OutputLayer:

    def __init__(self, number, activation=activations.Sigmoid):
        self.number = number
        self.activation = activation()

        self.weights = []
        self.biases = []

        self.previous = None

    def feedforward(self, a):
        return self.activation.evaluate(np.dot(self.weights, a) + self.biases)

    def bind(self, layer):
        layer.next = self
        self.previous = layer
