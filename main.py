from threading import Timer
import time
import numpy as np
from mnist import MNIST
from network import activations, costs, networks, layers, optimizers


def reshape(dataset):
    return [(normalize(img), reshape_output(label)) for img, label in zip(dataset[0], dataset[1])]


def reshape_output(label):
    output= np.zeros(10)
    output[label] = 1
    return output


def normalize(dataset):
    return dataset / 255


loader = MNIST(path='../dataset', return_type='numpy', gz=True)

training_set = loader.load_training()
test_set = loader.load_testing()

training_set = reshape(training_set)
test_set = reshape(test_set)

# for i in range(100, 110):
#     plt.imshow(np.reshape(test_set[i][0], (28, 28)), cmap='gray_r')
#     plt.show()
#

net3 = networks.NeuralNetwork(layers.InputLayer(784),
                              [layers.HiddenLayer(30, dropout=0.5)],
                              layers.OutputLayer(10))
net4 = networks.NeuralNetwork(layers.InputLayer(784),
                              [layers.HiddenLayer(100, activations.ReLU)],
                              layers.OutputLayer(10, activations.ReLU),
                              cost=costs.CrossEntropy)

net3.name = 'Net3'
net4.name = 'Net4'

optimizer = optimizers.StochasticGradientDescent(30, 10, 0.1, 5.0)


# Timer(0, net3.train_by_sgd, (training_set[:1000], 30, 10, 1, 5.0, test_set)).start()
# Timer(0, net4.train_by_sgd, (training_set[:1000], 30, 10, 0.1, 5.0, test_set)).start()


net4.train(optimizer, training_set[:1000], test_set)


# net4.save('test.json')
# print('training_data: {0}%'.format(net2.evaluate(training_set) / len(training_set) * 100))
# print('test_data: {0}%'.format(net2.evaluate(test_set) / len(test_set) * 100))

# test = np.arange(-100000, 100000)
# # test = np.reshape(test, (1000000, 1))
#
# start = time.clock()
# for i in range(1):
#     activations.LeakyReLU().evaluate(test)
# end = time.clock()
# print((end - start) * 1000)
#
# start = time.clock()
# for i in range(1):
#     np.maximum(test, 0.3 * test)
# end = time.clock()
# print((end - start) * 1000)

# utils.plot(activations.ELU().evaluate, -5, 5, 0.01)
# utils.plot(activations.LeakyReLU().evaluate, -5, 5, 0.01)
# utils.plot(activations.ThresholdedReLU().evaluate, -5, 5, 0.01)



