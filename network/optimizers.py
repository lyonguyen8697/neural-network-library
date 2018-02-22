import numpy as np


class GradientDescent:

    def __init__(self, learning_rate, weight_decay=0, momentum=0.9):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

    def optimize(self, network, training_set, callback=None):
        batch_size = len(training_set)

        for epoch in range(self.epochs):
            self.gradient_descent(network, training_set, batch_size)

            if callback is not None:
                callback(epoch)

    def gradient_descent(self, network, training_set, batch_size=None):
        if batch_size is None:
            batch_size = len(training_set)

        gradient_w = [np.zeros(layer.weights.shape) for layer in network.hidden_layers + [network.output_layer]]
        gradient_b = [np.zeros(layer.biases.shape) for layer in network.hidden_layers + [network.output_layer]]

        for inputs, outputs in training_set:
            delta_gradient_w, delta_gradient_b = self.backpropagation(network, inputs, outputs)
            gradient_w = [gradient + delta for gradient, delta in zip(gradient_w, delta_gradient_w)]
            gradient_b = [gradient + delta for gradient, delta in zip(gradient_b, delta_gradient_b)]

        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        update_w = np.zeros_like(gradient_w)
        update_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            update_w[index] = self.momentum * update_w[index] + self.learning_rate / len(training_set) * gradient_w[index]
            update_b[index] = self.momentum * update_b[index] + self.learning_rate / len(training_set) * gradient_b[index]

            layer.weights = weight_decay * layer.weights - update_w[index]
            layer.biases = layer.biases - update_b[index]

    @staticmethod
    def backpropagation(network, inputs, outputs):
        inputs = np.reshape(inputs, (network.input_layer.numbers, 1))
        outputs = np.reshape(outputs, (network.output_layer.numbers, 1))

        gradient_w = [np.zeros(layer.weights.shape) for layer in network.hidden_layers + [network.output_layer]]
        gradient_b = [np.zeros(layer.biases.shape) for layer in network.hidden_layers + [network.output_layer]]

        activation = network.input_layer.activate(inputs)
        activations = [activation]
        zs = []

        for layer in network.hidden_layers + [network.output_layer]:
            activation, z = layer.activate(activation)

            zs.append(z)
            activations.append(activation)

        delta = network.cost.delta(zs[-1], activations[-1], outputs, network.output_layer.activation)
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        gradient_b[-1] = delta

        for index, layer in enumerate(reversed(network.hidden_layers), 2):
            z = zs[-index]
            delta = np.dot(layer.next.weights.transpose(), delta) * layer.activation.derivative(z)

            gradient_w[-index] = np.dot(delta, activations[-index - 1].transpose())
            gradient_b[-index] = delta

        return gradient_w, gradient_b


class StochasticGradientDescent(GradientDescent):

    def __init__(self, epochs, mini_batch_size, learning_rate, weight_decay=0, momentum=0.9):
        GradientDescent.__init__(self, learning_rate, weight_decay, momentum)
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

    def optimize(self, network, training_set, callback=None):
        batch_size = len(training_set)

        for epoch in range(self.epochs):
            np.random.shuffle(training_set)
            mini_batches = self.separate_batch(training_set, self.mini_batch_size)

            for mini_batch in mini_batches:
                self.gradient_descent(network, mini_batch, batch_size)

            if callback is not None:
                callback(epoch)

    @staticmethod
    def separate_batch(dataset, mini_batch_size):
        return [dataset[k:k + mini_batch_size] for k in range(0, len(dataset), mini_batch_size)]
