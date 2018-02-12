import numpy as np


class MeanSquaredError:

    @staticmethod
    def evaluate(prediction, output):
        return 0.5 * np.linalg.norm(prediction - output) ** 2

    @staticmethod
    def delta(z, a, output, activation):
        return (a - output) * activation.derivative(z)


class CrossEntropy:

    @staticmethod
    def evaluate(prediction, output):
        return np.sum(np.nan_to_num(-output * np.log(prediction) - (1 - output) * np.log(1 - prediction)))

    @staticmethod
    def delta(z, a, output, activation):
        return a - output


class LogLikelihood:

    @staticmethod
    def evaluate(prediction, output):
        return -np.log(prediction[np.argmax(output)])

    @staticmethod
    def delta(z, a, output, activation):
        return a - output
