import numpy as np


class _Activation:

    def serialize(self):
        return {
            'name': self.__class__.__name__
        }

    @staticmethod
    def deserialize(o):
        pass

    def __call__(self, *args, **kwargs):
        return self


class Linear(_Activation):

    @staticmethod
    def evaluate(z):
        return z

    @staticmethod
    def derivative(z):
        return np.ones_like(z)


class Sigmoid(_Activation):

    @staticmethod
    def evaluate(z):
        e_z = np.exp(-(z + np.max(z)))
        return 1 / (1 + e_z)

    @staticmethod
    def derivative(z):
        sigmoid_z = Sigmoid.evaluate(z)
        return sigmoid_z * (1 - sigmoid_z)


class Tanh(_Activation):

    @staticmethod
    def evaluate(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z) ** 2


class ReLU(_Activation):

    @staticmethod
    def evaluate(z):
        return np.maximum(z, 0)

    @staticmethod
    def derivative(z):
        a = np.ones_like(z)
        a[z <= 0] = 0
        return a


class LeakyReLU(_Activation):

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def evaluate(self, z):
        return np.where(z >= 0, z, self.alpha * z)

    def derivative(self, z):
        return np.where(z >= 0, 1, self.alpha)

    def serialize(self):
        activation = _Activation.serialize()
        activation['alpha'] = self.alpha


class ELU(_Activation):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def evaluate(self, z):
        e_z = np.exp(z - np.max(z))
        return np.where(z >= 0, z, self.alpha * (e_z - 1))

    def derivative(self, z):
        e_z = np.exp(z - np.max(z))
        return np.where(z >= 0, 1, self.alpha * e_z)

    def serialize(self):
        activation = _Activation.serialize()
        activation['alpha'] = self.alpha


class ThresholdedReLU(_Activation):

    def __init__(self, theta=1.0):
        self.theta = theta

    def evaluate(self, z):
        return np.where(z > self.theta, z, 0)

    def derivative(self, z):
        return np.where(z > self.theta, 1, 0)

    def serialize(self):
        activation = _Activation.serialize()
        activation['theta'] = self.theta


class Softmax(_Activation):

    @staticmethod
    def evaluate(z):
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z)

    @staticmethod
    def derivative(z):
        softmax_z = Softmax.evaluate(z)
        return softmax_z * (1 - softmax_z)


class Softplus(_Activation):

    @staticmethod
    def evaluate(z):
        e_z = np.exp(z - np.max(z))
        return np.log(1 + e_z)

    @staticmethod
    def derivative(z):
        e_z = np.exp(-(z + np.max(z)))
        return 1 / (1 + e_z)
