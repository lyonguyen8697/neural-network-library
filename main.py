from threading import Timer
import numpy as np
from mnist import MNIST
from network import activations, costs, networks, layers, optimizers


def reshape_data(data):
    data = np.transpose(data)

    return normalize(data)


def reshape_labels(labels):
    output = np.zeros((10, len(labels)))
    for i in range(len(labels)):
        output[labels[i], i] = 1
    return output


def normalize(dataset):
    return dataset / 255


loader = MNIST(path='../dataset', return_type='numpy', gz=True)

training_set = loader.load_training()
test_set = loader.load_testing()

training_data = reshape_data(training_set[0])
training_labels = reshape_labels(training_set[1])
test_data = reshape_data(test_set[0])
test_labels = reshape_labels(test_set[1])

# for i in range(100, 110):
#     plt.imshow(np.reshape(test_set[i][0], (28, 28)), cmap='gray_r')
#     plt.show()
#

net3 = networks.NeuralNetwork(layers.InputLayer(784),
                              [
                                  layers.HiddenLayer(100, activations.ReLU),
                                  layers.HiddenLayer(100, activations.ReLU)
                              ],
                              layers.OutputLayer(10, activations.Sigmoid),
                              cost=costs.CrossEntropy)
net4 = networks.NeuralNetwork(layers.InputLayer(784),
                              [layers.HiddenLayer(100, activations.ReLU)],
                              layers.OutputLayer(10, activations.Sigmoid),
                              cost=costs.CrossEntropy)

net3.name = 'Net3'
net4.name = 'Net4'

# optimizer3 = optimizers.StochasticGradientDescent(30, 64, 1., 5.0)
optimizer3 = optimizers.StochasticGradientDescent(30, 64, 1., 5., nesterov=True)


Timer(0, net3.train, (optimizer3, (training_data, training_labels), (test_data, test_labels))).start()
# Timer(0, net4.train, (optimizer4, training_set, test_set)).start()


# net4.train(optimizer4, training_set[:1000], test_set)


# net4.save('test.json')
# print('training_data: {0}%'.format(net2.evaluate(training_set) / len(training_set) * 100))
# print('test_data: {0}%'.format(net2.evaluate(test_set) / len(test_set) * 100))



