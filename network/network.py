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
            layer.weights = np.random.randn(layer.number, layer.previous.number) / np.sqrt(layer.previous.number)
            layer.biases = np.random.randn(layer.number, 1)

    def feedforward(self, input):
        output = np.reshape(input, (self.input_layer.number, 1))

        start = time.clock()
        for layer in self.hidden_layers + [self.output_layer]:
            output = layer.feedforward(output)
        end = time.clock()
        # print((end - start) * 1000)
        return output

    def train_by_sgd(self, dataset, epochs, mini_batch_size, learning_rate, weight_decay=0, test_data=None):
        start = time.time()
        batch_size = len(dataset)

        for epoch in range(epochs):
            np.random.shuffle(dataset)
            mini_batches = [dataset[k:k + mini_batch_size] for k in range(0, batch_size, mini_batch_size)]

            for mini_batch in mini_batches:
                self.train(mini_batch, learning_rate, weight_decay, batch_size)

            if test_data:
                print("{0} Epoch {1}: {2} / {3}".format(self.name, epoch, self.evaluate(test_data), len(test_data)))
            else:
                print("{0} Epoch {1} complete".format(self.name, epoch))

        end = time.time()
        print('{0} finished in {1} sec'.format(self.name, end - start))
        self.testing(dataset, test_data)

    def train(self, dataset, learning_rate, weight_decay=0, batch_size=None):
        if not batch_size:
            batch_size = len(dataset)

        gradient_w = [np.zeros(layer.weights.shape) for layer in self.hidden_layers + [self.output_layer]]
        gradient_b = [np.zeros(layer.biases.shape) for layer in self.hidden_layers + [self.output_layer]]

        for input, output in dataset:
            delta_gradient_w, delta_gradient_b = self.backpropagation(input, output)
            gradient_w = [gradient + delta for gradient, delta in zip(gradient_w, delta_gradient_w)]
            gradient_b = [gradient + delta for gradient, delta in zip(gradient_b, delta_gradient_b)]

        weight_decay = (1 - learning_rate * weight_decay / batch_size)

        for index, layer in enumerate(self.hidden_layers + [self.output_layer]):
            layer.weights = weight_decay * layer.weights - learning_rate / len(dataset) * gradient_w[index]
            layer.biases = layer.biases - learning_rate / len(dataset) * gradient_b[index]

    def backpropagation(self, input, output):
        input = np.reshape(input, (self.input_layer.number, 1))
        output = np.reshape(output, (self.output_layer.number, 1))

        gradient_w = [np.zeros(layer.weights.shape) for layer in self.hidden_layers + [self.output_layer]]
        gradient_b = [np.zeros(layer.biases.shape) for layer in self.hidden_layers + [self.output_layer]]

        activation = self.input_layer.activate(input)
        activations = [activation]
        zs = []

        for layer in self.hidden_layers + [self.output_layer]:
            activation, z = layer.activate(activation)

            zs.append(z)
            activations.append(activation)

        delta = self.cost.delta(zs[-1], activations[-1], output, self.output_layer.activation)
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        gradient_b[-1] = delta

        for index, layer in enumerate(reversed(self.hidden_layers), 2):
            z = zs[-index]
            delta = np.dot(layer.next.weights.transpose(), delta) * layer.activation.derivative(z)

            gradient_w[-index] = np.dot(delta, activations[-index - 1].transpose())
            gradient_b[-index] = delta

        return gradient_w, gradient_b

    def evaluate(self, dataset):
        results = [(np.argmax(self.feedforward(input)), np.argmax(output)) for (input, output) in dataset]
        return sum(int(predict == output) for (predict, output) in results)

    def save_model(self, path):
        np.save(path, (self.sizes, self.weights, self.biases))

    def load_model(self, path):
        self.sizes, self.weights, self.biases = np.load(path)

    def testing(self, train_data, test_data=None):
        print('{0} Training data: {1}%'.format(self.name, self.evaluate(train_data) / len(train_data) * 100))
        if test_data:
            print('{0} Test data: {1}%'.format(self.name, self.evaluate(test_data) / len(test_data) * 100))