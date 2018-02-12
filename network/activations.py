import numpy as np


class Linear:

    @staticmethod
    def evaluate(z):
        return z

    @staticmethod
    def derivative(z):
        return np.ones_like(z)


class Sigmoid:

    @staticmethod
    def evaluate(z):
        e_z = np.exp(-(z + np.max(z)))
        return 1 / (1 + e_z)

    @staticmethod
    def derivative(z):
        sigmoid_z = Sigmoid.evaluate(z)
        return sigmoid_z * (1 - sigmoid_z)


class Tanh:

    @staticmethod
    def evaluate(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z) ** 2


class ReLU:

    @staticmethod
    def evaluate(z):
        return np.maximum(z, 0)

    @staticmethod
    def derivative(z):
        a = np.ones_like(z)
        a[z <= 0] = 0
        return a


class LeakyReLU:

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def evaluate(self, z):
        return np.where(z >= 0, z, self.alpha * z)

    def derivative(self, z):
        return np.where(z >= 0, 1, self.alpha)

    def __call__(self, *args, **kwargs):
        return self


class ELU:

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def evaluate(self, z):
        e_z = np.exp(z - np.max(z))
        return np.where(z >= 0, z, self.alpha * (e_z - 1))

    def derivative(self, z):
        e_z = np.exp(z - np.max(z))
        return np.where(z >= 0, 1, self.alpha * e_z)

    def __call__(self, *args, **kwargs):
        return self


class ThresholdedReLU:

    def __init__(self, theta=1.0):
        self.theta = theta

    def evaluate(self, z):
        return np.where(z > self.theta, z, 0)

    def derivative(self, z):
        return np.where(z > self.theta, 1, 0)

    def __call__(self, *args, **kwargs):
        return self


class Softmax:

    @staticmethod
    def evaluate(z):
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z)

    @staticmethod
    def derivative(z):
        softmax_z = Softmax.evaluate(z)
        return softmax_z * (1 - softmax_z)


class Softplus:

    @staticmethod
    def evaluate(z):
        e_z = np.exp(z - np.max(z))
        return np.log(1 + e_z)

    @staticmethod
    def derivative(z):
        e_z = np.exp(-(z + np.max(z)))
        return 1 / (1 + e_z)
