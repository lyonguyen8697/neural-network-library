import numpy as np


class GradientDescent:

    def __init__(self, epochs, learning_rate, weight_decay=0.0, momentum=0.9, nesterov=False):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

    def optimize(self, network, training_set, callback=None):
        batch_size = training_set[0].shape[1]

        for epoch in range(self.epochs):
            self.gradient_descent(network, training_set, batch_size)

            if callback is not None:
                callback(epoch)

    def gradient_descent(self, network, training_set, batch_size=None):
        if batch_size is None:
            batch_size = training_set[0].shape[1]

        gradient_w = [np.zeros(layer.weights.shape) for layer in network.hidden_layers + [network.output_layer]]
        gradient_b = [np.zeros(layer.biases.shape) for layer in network.hidden_layers + [network.output_layer]]

        data, labels = training_set

        delta_gradient_w, delta_gradient_b = self.backpropagation(network, data, labels)
        gradient_w = [gradient + delta for gradient, delta in zip(gradient_w, delta_gradient_w)]
        gradient_b = [gradient + delta for gradient, delta in zip(gradient_b, delta_gradient_b)]

        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        velocity_w = np.zeros_like(gradient_w)
        velocity_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):

            grad_update_w = self.learning_rate / batch_size * gradient_w[index]
            grad_update_b = self.learning_rate / batch_size * gradient_b[index]

            velocity_w[index] = self.momentum * velocity_w[index] - grad_update_w
            velocity_b[index] = self.momentum * velocity_b[index] - grad_update_b

            if self.nesterov:
                update_w = self.momentum * velocity_w[index] - grad_update_w
                update_b = self.momentum * velocity_b[index] - grad_update_b
            else:
                update_w = velocity_w[index]
                update_b = velocity_b[index]

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b

    @staticmethod
    def backpropagation(network, data, labels):
        gradient_w = [np.zeros(layer.weights.shape) for layer in network.hidden_layers + [network.output_layer]]
        gradient_b = [np.zeros(layer.biases.shape) for layer in network.hidden_layers + [network.output_layer]]

        activation = network.input_layer.activate(data)
        activations = [activation]
        zs = []

        for layer in network.hidden_layers + [network.output_layer]:
            activation, z = layer.activate(activation)

            zs.append(z)
            activations.append(activation)

        delta = network.cost.delta(zs[-1], activations[-1], labels, network.output_layer.activation)
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        gradient_b[-1] = np.sum(delta, axis=1, keepdims=True)

        for index, layer in enumerate(reversed(network.hidden_layers), 2):
            z = zs[-index]
            delta = np.dot(layer.next.weights.transpose(), delta) * layer.activation.derivative(z)

            gradient_w[-index] = np.dot(delta, activations[-index - 1].transpose())
            gradient_b[-index] = np.sum(delta, axis=1, keepdims=True)

        return gradient_w, gradient_b


class StochasticGradientDescent(GradientDescent):

    def __init__(self, epochs, mini_batch_size, learning_rate, weight_decay=0, momentum=0.9, nesterov=False):
        GradientDescent.__init__(self, epochs, learning_rate, weight_decay, momentum, nesterov)
        self.mini_batch_size = mini_batch_size

    def optimize(self, network, training_set, callback=None):
        batch_size = training_set[0].shape[1]

        for epoch in range(self.epochs):
            state = np.random.get_state()
            np.random.shuffle(training_set[0].T)
            np.random.set_state(state)
            np.random.shuffle(training_set[1].T)

            mini_batches = self.separate_batch(training_set, self.mini_batch_size)

            for mini_batch in mini_batches:
                self.gradient_descent(network, mini_batch, batch_size)

            if callback is not None:
                callback(epoch)

    @staticmethod
    def separate_batch(dataset, mini_batch_size):
        return [(dataset[0][:, k:k + mini_batch_size], dataset[1][:, k:k + mini_batch_size]) for k in range(0, dataset[0].shape[1], mini_batch_size)]
