import numpy as np
import matplotlib.pyplot as plt
from Constants import *


class NeuralNetwork:
    def __init__(self):
        # setup
        self.biases = []
        self.create_biases()

        self.weights = []
        self.create_weights()

    def train_with_progress_indicators(self, images, labels, test_images, test_labels):
        print('Training starting...')
        costs_while_training = []
        accuracies_while_training = []

        mini_batches = self.make_mini_batches(images, labels)

        # output training statistics every n batches
        for i in range(EPOCHS):
            np.random.shuffle(mini_batches)
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch[0], mini_batch[1])

            # test and display results
            current_cost = self.get_test_cost(test_images, test_labels)
            current_accuracy = self.get_test_accuracy(test_images, test_labels)

            print("Current cost: ", current_cost)
            print("Current accuracy", current_accuracy * 100, "%")

            costs_while_training.append(current_cost)
            accuracies_while_training.append(current_accuracy)

            # display progress as number of epochs completed
            progress = (i / EPOCHS)
            print("Training progress: ", int(100 * progress), "%", "\n")

        print('Training finished!')
        # plot cost and accuracy against training examples
        plt.plot(costs_while_training, linestyle='dotted')
        plt.plot(accuracies_while_training, linestyle='solid')
        plt.show()

    def make_mini_batches(self, images, labels):
        # make list of mini batches
        # each mini batch is a tuple of lists ([images], [labels])
        mini_batches = []
        for i in range(0, len(images), BATCH_SIZE):
            mini_batches.append((images[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE]))
        return mini_batches

    def train_mini_batch(self, images, labels):
        weight_changes_sum = self.create_weight_changes()
        bias_changes_sum = self.create_bias_changes()

        # find average change to weights and biases needed
        for i in range(BATCH_SIZE):
            # backpropogate
            weight_changes, bias_changes = self.backpropogate(labels[i], images[i])

            # average over the requested changes by each image
            for layer in range(NUM_LAYERS - 1):
                weight_changes_sum[layer] += weight_changes[layer]
                bias_changes_sum[layer] += bias_changes[layer]

        # subtract changes to weights
        for layer in range(NUM_LAYERS - 1):
            self.weights[layer] = self.weights[layer] - (LEARNING_RATE / BATCH_SIZE) * weight_changes_sum[layer]
            self.biases[layer] = self.biases[layer] - (LEARNING_RATE / BATCH_SIZE) * bias_changes_sum[layer]

    def get_test_accuracy(self, images, labels):
        # return average accuracy (whether it has correctly predicted) of given images
        # returns number between 0 and 1
        correct = 0
        for i in range(len(images)):
            # feed forward and get predicted value
            predicted_digit = self.calculate_predicted_digit(self.feed_forward(images[i]))

            if labels[i] == predicted_digit:
                correct += 1
        return correct / len(images)

    def get_test_cost(self, images, labels):
        # return average cost of given images
        cost = 0

        for i in range(len(images)):
            cost += self.calculate_cost(labels[i], self.feed_forward(images[i]))
        return cost / len(images)

    def create_weights(self):
        # create array of weights for each layer, all set to random numbers
        for i in range(NUM_LAYERS - 1):
            # create 2D array of random numbers from normal distribution
            layer_weights = np.random.randn(NEURONS_IN_LAYER[i + 1], NEURONS_IN_LAYER[i])
            self.weights.append(layer_weights)
            # for ith layer:
            # [[ ... weights for first neuron in ith layer  ... ],
            # [ ... weights for second neuron in ith layer ... ],
            # ... ]
            # so should be same number of columns as neurons in last layer
            # and same number of rows as neurons in this layer

            # first layer doesn't have weight

    def create_weight_changes(self):
        # create zero filled list of same shape as self.weights
        weight_changes = []
        for i in range(NUM_LAYERS - 1):
            layer_weights = np.zeros((NEURONS_IN_LAYER[i + 1], NEURONS_IN_LAYER[i]))
            weight_changes.append(layer_weights)
        return weight_changes

    def create_biases(self):
        for i in range(NUM_LAYERS - 1):
            # create array of random numbers from normal distribution
            layer_biases = np.random.randn(NEURONS_IN_LAYER[i + 1])
            self.biases.append(layer_biases)
            # for ith layer:
            # each neuron has one bias
            # [ bias for first neuron in ith layer,
            #   bias for second neuron in ith layer,
            # ... ]
            # first layer doesn't have bias

    def create_bias_changes(self):
        # create zero filled list of same shape as self.biases
        bias_changes = []
        for i in range(NUM_LAYERS - 1):
            layer_biases = np.zeros((NEURONS_IN_LAYER[i + 1]))
            bias_changes.append(layer_biases)
        return bias_changes

    def feed_forward(self, activations):
        a = activations
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, a) + b
            a = self.activation_function(z)
        return a

    def backpropogate(self, expected_digit, activations):
        # feed forward
        a = [activations]
        z = []
        for b, w in zip(self.biases, self.weights):
            z.append(np.matmul(w, a[-1]) + b)
            a.append(self.activation_function(z[-1]))

        # initialise weight_changes and bias_changes
        weight_changes = self.create_weight_changes()
        bias_changes = self.create_bias_changes()

        # backpropogate

        # activation_changes[l] is dc/da^l for the desired layer l
        expected_final_layer = self.calculate_expected_last_layer(expected_digit)
        activation_changes = self.cost_derivative(expected_final_layer, a[NUM_LAYERS - 1])

        for l in range(1, NUM_LAYERS)[::-1]:

            dc_by_dz = activation_changes * self.da_by_dz(a[l])

            weight_changes[l - 1] = np.outer(dc_by_dz, self.dz_by_dw(a[l - 1]))

            # calculate change to bias needed and add to requested change
            bias_changes[l - 1] = dc_by_dz * self.dz_by_db()

            # calculate change to activations needed
            activation_changes = np.matmul(dc_by_dz, self.dz_by_da(self.weights[l - 1]))

        return weight_changes, bias_changes

    @staticmethod
    def activation_function(weighted_input):
        # watch out for possible overflows for numbers > +- pow(2, 8)
        activation = 1 / (1 + np.exp(-weighted_input))
        return activation

    @staticmethod
    def calculate_cost(expected_output, average_activations):
        cost = 0
        for i in range(NEURONS_IN_LAYER[NUM_LAYERS - 1]):
            if i == expected_output:
                cost += (average_activations[i] - 1) ** 2
            else:
                cost += (average_activations[i]) ** 2
        return cost

    @staticmethod
    def calculate_expected_last_layer(label):
        # returns the expected final layer in network, given the label
        expected_last_layer = np.zeros(NEURONS_IN_LAYER[NUM_LAYERS - 1])
        expected_last_layer[label] = 1.0
        return expected_last_layer

    @staticmethod
    def calculate_predicted_digit(last_layer_activations):
        # work out neuron in last layer with the greatest activation
        # the index of this neuron is the predicted number
        max_activation = 0
        max_activation_index = 0
        for i in range(NEURONS_IN_LAYER[NUM_LAYERS - 1]):
            if last_layer_activations[i] > max_activation:
                max_activation = last_layer_activations[i]
                max_activation_index = i

        return max_activation_index

    @staticmethod
    def cost_derivative(expected_output, activation):
        # dc/da
        derivative = activation - expected_output
        return derivative

    @staticmethod
    def da_by_dz(activation):
        # da/dz
        # derivative of sigmoid
        return activation * (1 - activation)

    @staticmethod
    def dz_by_dw(input_activation):
        return input_activation

    @staticmethod
    def dz_by_db():
        return 1

    @staticmethod
    def dz_by_da(weight):
        return weight
