import json
import numpy as np
from network import costs
import time


class NeuralNetwork:

    def __init__(self, input_layer, hidden_layers, output_layer, cost=costs.CrossEntropy):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.bind_layers()

        self.cost = cost

        self.init_parameters()

    def bind_layers(self):
        layers = [self.input_layer] + self.hidden_layers + [self.output_layer]

        for i in range(1, len(layers)):
            layers[i].bind(layers[i - 1])

    def init_parameters(self):
        for layer in self.hidden_layers + [self.output_layer]:
            layer.weights = np.random.randn(layer.numbers, layer.previous.numbers) / np.sqrt(layer.previous.numbers)
            layer.biases = np.random.randn(layer.numbers, 1)

    def feedforward(self, inputs):
        outputs = np.reshape(inputs, (self.input_layer.numbers, 1))

        for layer in self.hidden_layers + [self.output_layer]:
            outputs = layer.feedforward(outputs)

        return outputs

    def train(self, optimizer, training_set, test_set=None):
        def callback(epoch):
            if test_set is not None:
                print('{0} Epoch {1}: {2}/{3}'.format(self.name, epoch, self.evaluate(test_set), len(test_set)))
            else:
                print('{0} Epoch {1} complete'.format(self.name, epoch))

        start = time.time()

        optimizer.optimize(self, training_set, callback)

        end = time.time()

        print('Training {0} finished in {1:.2f} second(s)'.format(self.name, end - start))

    def evaluate(self, dataset):
        results = [(np.argmax(self.feedforward(inputs)), np.argmax(outputs)) for (inputs, outputs) in dataset]
        return sum(int(predict == output) for (predict, output) in results)

    def serialize(self):
        return {
            'cost': self.cost.serialize(),
            'last train': 'None',
            'layers': {
                'input': self.input_layer.serialize(),
                'hidden': [layer.serialize() for layer in self.hidden_layers],
                'output': self.output_layer.serialize()
            }
        }

    def save(self, path):
        model = self.serialize()

        with open(path, 'w') as f:
            json.dump(model, f)

    def load(self, path):
        pass

    def testing(self, train_data, test_data=None):
        print('{0} Training data: {1}%'.format(self.name, self.evaluate(train_data) / len(train_data) * 100))
        if test_data:
            print('{0} Test data: {1}%'.format(self.name, self.evaluate(test_data) / len(test_data) * 100))