import numpy as np


class GradientDescentOptimizer:

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.weight_decay = weight_decay

        self.parameters_updated_count = 0

    def optimize(self, network, training_set, callback=None):
        batch_size = training_set[0].shape[1]
        if self.mini_batch_size is None:
            self.mini_batch_size = batch_size

        self.parameters_updated_count = 0

        for epoch in range(self.epochs):
            self.shuffle_batch(training_set)

            mini_batches = self.separate_batch(training_set, self.mini_batch_size)

            for mini_batch in mini_batches:
                self.gradient_descent(network, mini_batch, batch_size)

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

        self.parameters_updated_count += 1
        self.update_parameters(network, gradient_w, gradient_b, batch_size)

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            update_w = - self.learning_rate / batch_size * gradient_w[index]
            update_b = - self.learning_rate / batch_size * gradient_b[index]

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

    @staticmethod
    def shuffle_batch(dataset):
        state = np.random.get_state()
        np.random.shuffle(dataset[0].T)
        np.random.set_state(state)
        np.random.shuffle(dataset[1].T)

    @staticmethod
    def separate_batch(dataset, mini_batch_size):
        return [(dataset[0][:, k:k + mini_batch_size], dataset[1][:, k:k + mini_batch_size]) for k in
                range(0, dataset[0].shape[1], mini_batch_size)]


class MomentumOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, gamma=0.9):
        GradientDescentOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay)
        self.gamma = gamma

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        momentum_w = np.zeros_like(gradient_w)
        momentum_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            momentum_w[index] = self.calculate_momentum(gradient_w[index], momentum_w[index], self.gamma)
            momentum_b[index] = self.calculate_momentum(gradient_b[index], momentum_b[index], self.gamma)

            update_w = - self.learning_rate / batch_size * momentum_w[index]
            update_b = - self.learning_rate / batch_size * momentum_b[index]

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b

    @staticmethod
    def calculate_momentum(pre_momentum, gradient, momentum):
        if momentum > 0.0:
            return momentum * pre_momentum + gradient
        else:
            return gradient


class NAGOptimizer(MomentumOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, gamma=0.9):
        MomentumOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay, gamma)

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        momentum_w = np.zeros_like(gradient_w)
        momentum_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            momentum_w[index] = self.calculate_momentum(gradient_w[index], momentum_w[index], self.gamma)
            momentum_b[index] = self.calculate_momentum(gradient_b[index], momentum_b[index], self.gamma)

            nesterov_w = self.calculate_momentum(gradient_w[index], momentum_w[index], self.gamma)
            nesterov_b = self.calculate_momentum(gradient_b[index], momentum_b[index], self.gamma)

            update_w = - self.learning_rate / batch_size * nesterov_w
            update_b = - self.learning_rate / batch_size * nesterov_b

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b


class AdagradOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, epsilon=1e-8):
        GradientDescentOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay)
        self.epsilon = epsilon

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        adagrad_w = np.zeros_like(gradient_w)
        adagrad_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            adagrad_w[index] += gradient_w[index]**2
            adagrad_b[index] += gradient_b[index]**2

            update_w = - self.learning_rate / (batch_size * np.sqrt(adagrad_w[index] + self.epsilon)) * gradient_w[index]
            update_b = - self.learning_rate / (batch_size * np.sqrt(adagrad_b[index] + self.epsilon)) * gradient_b[index]

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b


class AdadeltaOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, gamma=0.9, epsilon=1e-8):
        GradientDescentOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay)
        self.gamma = gamma
        self.epsilon = epsilon

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        adadelta_gradient_w = np.zeros_like(gradient_w)
        adadelta_gradient_b = np.zeros_like(gradient_b)

        adadelta_update_w = np.ones_like(gradient_w)
        adadelta_update_b = np.ones_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            adadelta_gradient_w[index] = self.calculate_ema(adadelta_gradient_w[index], gradient_w[index], self.gamma)
            adadelta_gradient_b[index] = self.calculate_ema(adadelta_gradient_b[index], gradient_b[index], self.gamma)

            update_w = - np.sqrt(adadelta_update_w[index] + self.epsilon) / (batch_size * np.sqrt(adadelta_gradient_w[index] + self.epsilon)) * gradient_w[index]
            update_b = - np.sqrt(adadelta_update_b[index] + self.epsilon) / (batch_size * np.sqrt(adadelta_gradient_b[index] + self.epsilon)) * gradient_b[index]

            adadelta_update_w[index] = self.calculate_ema(adadelta_update_w[index], update_w, self.gamma)
            adadelta_update_b[index] = self.calculate_ema(adadelta_update_b[index], update_b, self.gamma)

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b

    @staticmethod
    def calculate_ema(pre_ema, grad, gamma):
        return gamma * pre_ema + (1 - gamma) * grad**2


class RMSpropOptimizer(GradientDescentOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, beta=0.999, epsilon=1e-8):
        GradientDescentOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay)
        self.beta = beta
        self.epsilon = epsilon

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        RMSprop_w = np.zeros_like(gradient_w)
        RMSprop_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            RMSprop_w[index] = self.calculate_rmsprop(RMSprop_w[index], gradient_w[index], self.beta)
            RMSprop_b[index] = self.calculate_rmsprop(RMSprop_b[index], gradient_b[index], self.beta)

            update_w = - self.learning_rate / (batch_size * np.sqrt(RMSprop_w[index] + self.epsilon)) * gradient_w[index]
            update_b = - self.learning_rate / (batch_size * np.sqrt(RMSprop_b[index] + self.epsilon)) * gradient_b[index]

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b

    @staticmethod
    def calculate_rmsprop(pre_RMSprop, gradient, beta):
        return beta * pre_RMSprop + (1 - beta) * gradient ** 2


class AdamOptimizer(MomentumOptimizer, RMSpropOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        GradientDescentOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        momentum_w = np.zeros_like(gradient_w)
        momentum_b = np.zeros_like(gradient_b)

        rmsprop_w = np.zeros_like(gradient_w)
        rmsprop_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            momentum_w[index] = self.calculate_momentum(momentum_w[index], gradient_w[index], self.beta_1)
            momentum_b[index] = self.calculate_momentum(momentum_b[index], gradient_b[index], self.beta_1)

            momentum_hat_w = self.bias_correction(momentum_w[index], self.beta_1, self.parameters_updated_count)
            momentum_hat_b = self.bias_correction(momentum_b[index], self.beta_1, self.parameters_updated_count)

            rmsprop_w[index] = self.calculate_rmsprop(rmsprop_w[index], gradient_w[index], self.beta_2)
            rmsprop_b[index] = self.calculate_rmsprop(rmsprop_b[index], gradient_b[index], self.beta_2)

            rmsprop_hat_w = self.bias_correction(rmsprop_w[index], self.beta_2, self.parameters_updated_count)
            rmsprop_hat_b = self.bias_correction(rmsprop_b[index], self.beta_2, self.parameters_updated_count)

            update_w = - self.learning_rate / (batch_size * (np.sqrt(rmsprop_hat_w) + self.epsilon)) * momentum_hat_w
            update_b = - self.learning_rate / (batch_size * (np.sqrt(rmsprop_hat_b) + self.epsilon)) * momentum_hat_b

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b

    @staticmethod
    def calculate_momentum(pre_momentum, gradient, beta):
        return beta * pre_momentum + (1 - beta) * gradient

    @staticmethod
    def bias_correction(m, beta, t):
        return m / (1 - beta**t)


class AdamMaxOptimizer(AdamOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, beta_1=0.9, beta_2=0.999):
        AdamOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay, beta_1, beta_2, None)

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        momentum_w = np.zeros_like(gradient_w)
        momentum_b = np.zeros_like(gradient_b)

        adammax_w = np.ones_like(gradient_w)
        adammax_b = np.ones_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            momentum_w[index] = self.calculate_momentum(momentum_w[index], gradient_w[index], self.beta_1)
            momentum_b[index] = self.calculate_momentum(momentum_b[index], gradient_b[index], self.beta_1)

            momentum_hat_w = self.bias_correction(momentum_w[index], self.beta_1, self.parameters_updated_count)
            momentum_hat_b = self.bias_correction(momentum_b[index], self.beta_1, self.parameters_updated_count)

            adammax_w[index] = self.calculate_adammax(adammax_w[index], gradient_w[index], self.beta_2)
            adammax_b[index] = self.calculate_adammax(adammax_b[index], gradient_b[index], self.beta_2)

            update_w = - self.learning_rate / (batch_size * adammax_w[index]) * momentum_hat_w
            update_b = - self.learning_rate / (batch_size * adammax_b[index]) * momentum_hat_b

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b

    @staticmethod
    def calculate_adammax(pre_adammax, gradient, beta):
        return np.maximum(beta * pre_adammax, np.abs(gradient))


class NadamOptimizer(NAGOptimizer, AdamOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, beta_1=0.0, beta_2=0.999, epsilon=1e-8):
        AdamOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay, beta_1, beta_2, epsilon)

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        momentum_w = np.zeros_like(gradient_w)
        momentum_b = np.zeros_like(gradient_b)

        rmsprop_w = np.zeros_like(gradient_w)
        rmsprop_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            momentum_w[index] = self.calculate_momentum(momentum_w[index], gradient_w[index], self.beta_1)
            momentum_b[index] = self.calculate_momentum(momentum_b[index], gradient_b[index], self.beta_1)

            momentum_hat_w = self.bias_correction(momentum_w[index], self.beta_1, self.parameters_updated_count)
            momentum_hat_b = self.bias_correction(momentum_b[index], self.beta_1, self.parameters_updated_count)

            rmsprop_w[index] = self.calculate_rmsprop(rmsprop_w[index], gradient_w[index], self.beta_2)
            rmsprop_b[index] = self.calculate_rmsprop(rmsprop_b[index], gradient_b[index], self.beta_2)

            rmsprop_hat_w = self.bias_correction(rmsprop_w[index], self.beta_2, self.parameters_updated_count)
            rmsprop_hat_b = self.bias_correction(rmsprop_b[index], self.beta_2, self.parameters_updated_count)

            update_w = - self.learning_rate / (batch_size * (np.sqrt(rmsprop_hat_w) + self.epsilon)) * \
                       (self.beta_1 * momentum_hat_w + (1 - self.beta_1) * gradient_w[index] / (1 - self.beta_1**self.parameters_updated_count))
            update_b = - self.learning_rate / (batch_size * (np.sqrt(rmsprop_hat_b) + self.epsilon)) * \
                       (self.beta_1 * momentum_hat_b + (1 - self.beta_1) * gradient_b[index] / (1 - self.beta_1**self.parameters_updated_count))

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b


class AMSGradOptimizer(AdamOptimizer):

    def __init__(self, learning_rate, epochs, mini_batch_size, weight_decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        AdamOptimizer.__init__(self, learning_rate, epochs, mini_batch_size, weight_decay, beta_1, beta_2, epsilon)

    def update_parameters(self, network, gradient_w, gradient_b, batch_size):
        weight_decay = (1 - self.learning_rate * self.weight_decay / batch_size)

        momentum_w = np.zeros_like(gradient_w)
        momentum_b = np.zeros_like(gradient_b)

        rmsprop_w = np.zeros_like(gradient_w)
        rmsprop_b = np.zeros_like(gradient_b)

        amsgrad_w = np.zeros_like(gradient_w)
        amsgrad_b = np.zeros_like(gradient_b)

        for index, layer in enumerate(network.hidden_layers + [network.output_layer]):
            momentum_w[index] = self.calculate_momentum(momentum_w[index], gradient_w[index], self.beta_1)
            momentum_b[index] = self.calculate_momentum(momentum_b[index], gradient_b[index], self.beta_1)

            rmsprop_w[index] = self.calculate_rmsprop(rmsprop_w[index], gradient_w[index], self.beta_2)
            rmsprop_b[index] = self.calculate_rmsprop(rmsprop_b[index], gradient_b[index], self.beta_2)

            amsgrad_w[index] = self.calculate_amsgrad(amsgrad_w[index], rmsprop_w[index])
            amsgrad_b[index] = self.calculate_amsgrad(amsgrad_b[index], rmsprop_b[index])

            update_w = - self.learning_rate / (batch_size * (np.sqrt(amsgrad_w[index]) + self.epsilon)) * momentum_w[index]
            update_b = - self.learning_rate / (batch_size * (np.sqrt(amsgrad_b[index]) + self.epsilon)) * momentum_b[index]

            layer.weights = weight_decay * layer.weights + update_w
            layer.biases = layer.biases + update_b

    @staticmethod
    def calculate_amsgrad(pre_amsgrad, rmsprop):
        return np.maximum(pre_amsgrad, rmsprop)
