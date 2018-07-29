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

net1 = networks.NeuralNetwork(layers.InputLayer(784),
                              [
                                  layers.HiddenLayer(100, activations.ReLU)
                              ],
                              layers.OutputLayer(10, activations.Sigmoid),
                              cost=costs.CrossEntropy)
net2 = networks.NeuralNetwork(layers.InputLayer(784),
                              [layers.HiddenLayer(100, activations.ReLU)],
                              layers.OutputLayer(10, activations.Sigmoid),
                              cost=costs.CrossEntropy)

net1.name = 'Net1'
net2.name = 'Net2'

optimizer1 = optimizers.GradientDescentOptimizer(1., 30, 64, 5.0)
optimizer2 = optimizers.MomentumOptimizer(2., 30, 64, 5.0, gamma=0.9)
optimizer3 = optimizers.NAGOptimizer(2., 30, 64, 5.0, gamma=0.9)
optimizer4 = optimizers.AdagradOptimizer(2., 30, 64, 5.0)
optimizer5 = optimizers.AdadeltaOptimizer(2., 30, 64, 5.0, gamma=0.9)
optimizer6 = optimizers.RMSpropOptimizer(2., 30, 64, 5.0, beta=0.999)
optimizer7 = optimizers.AdamOptimizer(2., 30, 64, 5.0, beta_1=0.9, beta_2=0.999)
optimizer8 = optimizers.AdamMaxOptimizer(2., 30, 64, 5.0, beta_1=0.9, beta_2=0.999)
optimizer9 = optimizers.NadamOptimizer(2., 30, 64, 5.0, beta_1=0.9, beta_2=0.999)
optimizer10 = optimizers.AMSGradOptimizer(2., 30, 64, 5.0, beta_1=0.9, beta_2=0.999)


net1.train(optimizer6, (training_data, training_labels), (test_data, test_labels))

print('Training_data: {0}%'.format(net1.evaluate((training_data, training_labels)) / training_data.shape[1] * 100))
print('Test_data: {0}%'.format(net1.evaluate((test_data, test_labels)) / test_data.shape[1] * 100))

# Timer(0, net1.train, (optimizer1, (training_data, training_labels), (test_data, test_labels))).start()






