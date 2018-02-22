import numpy as np


class _CostFunction:

    @staticmethod
    def serialize():
        return _CostFunction.__class__.__name__


class MeanSquaredError(_CostFunction):

    @staticmethod
    def evaluate(prediction, output):
        return 0.5 * np.linalg.norm(prediction - output) ** 2

    @staticmethod
    def delta(z, a, output, activation):
        return (a - output) * activation.derivative(z)


class CrossEntropy(_CostFunction):

    @staticmethod
    def evaluate(prediction, output):
        return np.sum(np.nan_to_num(-output * np.log(prediction) - (1 - output) * np.log(1 - prediction)))

    @staticmethod
    def delta(z, a, output, activation):
        return a - output


class LogLikelihood(_CostFunction):

    @staticmethod
    def evaluate(prediction, output):
        return -np.log(prediction[np.argmax(output)])

    @staticmethod
    def delta(z, a, output, activation):
        return a - output
