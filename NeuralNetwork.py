import numpy as np
import matplotlib.pyplot as plt
from Constants import *


class CostFunction:
    # abstract class to be overridden by any child class
    @staticmethod
    def calculate_cost(expected_digit, final_layer_activations):
        pass

    @staticmethod
    def calculate_cost_derivative(expected_activations, final_layer_activations):
        # calculates partial derivative with respect to a (∂c/∂a)
        pass


class Quadratic(CostFunction):
    @staticmethod
    def calculate_cost(expected_digit, final_layer_activations):
        # only for 1 training example not mini batch
        cost = 0
        for i in range(NEURONS_IN_LAYER[NUM_LAYERS - 1]):
            if i == expected_digit:
                cost += (final_layer_activations[i] - 1) ** 2
            else:
                cost += (final_layer_activations[i]) ** 2
        return cost

    @staticmethod
    def calculate_cost_derivative(expected_activations, final_layer_activations):
        # dc/da
        derivative = final_layer_activations - expected_activations
        return derivative


class CrossEntropy(CostFunction):
    @staticmethod
    def calculate_cost(expected_digit, final_layer_activations):
        expected_activations = calculate_expected_last_layer(expected_digit)
        # only for 1 training example not mini batch
        # np.log() is natural logarithm ln()
        cost_sum = 0
        for y, a in zip(expected_activations, final_layer_activations):
            cost_sum += (np.nan_to_num(y*np.log(a) + (1 - y) * np.log(1 - a)))

        cost = (-1 * cost_sum) / NEURONS_IN_LAYER[NUM_LAYERS - 1]

        return cost

    @staticmethod
    def calculate_cost_derivative(expected_activations, final_layer_activations):
        # dc/da
        y = expected_activations
        a = final_layer_activations
        derivative = ((1 - y) / (1 - a)) - (y / a)
        return derivative


class NeuralNetwork:
    def __init__(self):
        # setup
        self.biases = []
        self.create_biases()

        self.weights = []
        if WEIGHT_INITIALISATION.lower() == "large":
            self.create_large_weights()
        elif WEIGHT_INITIALISATION.lower() == "small":
            self.create_small_weights()

        self.cost = None
        if COST_FUNCTION.lower() == "cross entropy":
            self.cost = CrossEntropy()
        elif COST_FUNCTION.lower() == "quadratic":
            self.cost = Quadratic()

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

            print("Current cost: ", np.round(current_cost, 5))
            print("Current accuracy", np.round(current_accuracy * 100, 2), "%")

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

    def train_and_show_results_no_improvement_in_n(self, images, labels, test_images, test_labels):
        print('Training starting...')
        unregularised_cost_while_training = []
        regularisation_cost_while_training = []
        total_cost_while_training = []
        training_accuracies_while_training = []
        testing_accuracies_while_training = []

        mini_batches = self.make_mini_batches(images, labels)

        max_accuracy = 0
        epochs_since_improvement = 0
        total_epochs = 0
        accuracy_still_improving = True

        while accuracy_still_improving:
            np.random.shuffle(mini_batches)
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch[0], mini_batch[1])

            total_epochs += 1

            # test and display results
            current_cost = self.get_test_cost(test_images, test_labels)
            current_accuracy = self.get_test_accuracy(test_images, test_labels)

            unregularised_cost_while_training.append(self.get_normal_cost_term(images, labels))
            regularisation_cost_while_training.append(self.get_regularisation_cost_term(len(test_images)))

            training_accuracies_while_training.append(self.get_test_accuracy(images, labels))

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            accuracy_still_improving = (epochs_since_improvement < NO_IMPROVEMENT_EPOCH_LIMIT)

            print("Current cost: ", np.round(current_cost, 5))
            print("Current accuracy", np.round(current_accuracy * 100, 2), "%")

            total_cost_while_training.append(current_cost)
            testing_accuracies_while_training.append(current_accuracy)

            # display progress
            print("Epochs completed: ", total_epochs)
            print("Epochs since improvement: ", epochs_since_improvement)
            print("Best result: ", np.round(max_accuracy * 100, 2), "%", "\n")

        print('Training finished!')

        fig, ax = plt.subplots()

        ax.set_xlim(0, total_epochs - 1)
        ax.set_ylim(0, max(1, np.max(total_cost_while_training)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("Epochs")

        # plot cost and accuracy against number of epochs
        plt.plot(unregularised_cost_while_training, linestyle='dashed', color="blue", label="non-regularisation cost")
        plt.plot(regularisation_cost_while_training, linestyle='dashed', color="red", label="regularisation cost")
        plt.plot(total_cost_while_training, linestyle='dashed', color="purple", label="total cost")

        plt.plot(testing_accuracies_while_training, linestyle='solid', color="green", label="testing data accuracy (0-1)")
        plt.plot(training_accuracies_while_training, linestyle='solid', color="orange", label="training data accuracy (0-1)")

        plt.legend()
        plt.show()

    def make_mini_batches(self, images, labels):
        # make list of mini batches
        # each mini batch is a tuple of lists ([images], [labels])

        # if images / BATCH_SIZE is not an integer, there will be fewer images in mini batches than total images
        # and each remaining mini batch is of BATCH_SIZE

        mini_batches = []
        for mini_batch_index in range(int(len(images)/BATCH_SIZE)):
            index = mini_batch_index * BATCH_SIZE
            mini_batches.append((images[index:index + BATCH_SIZE], labels[index:index + BATCH_SIZE]))
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
            weight_decay = 1 - LEARNING_RATE * (REGULARISATION_PARAMETER / TRAINING_DATA_SIZE)
            self.weights[layer] = weight_decay * self.weights[layer] - (LEARNING_RATE / BATCH_SIZE) * weight_changes_sum[layer]
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

    def create_large_weights(self):
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

    def create_small_weights(self):
        # does same thing as create_large_weights,
        # but reduces the standard deviation of weights so the sum (z) is smaller and neuron less likely to saturate

        for i in range(NUM_LAYERS - 1):
            # create 2D array of random numbers from normal distribution
            stand_dev = 1/(NEURONS_IN_LAYER[i]**0.5)
            layer_weights = np.random.normal(0, stand_dev, (NEURONS_IN_LAYER[i + 1], NEURONS_IN_LAYER[i]))
            self.weights.append(layer_weights)

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
            a = self.calculate_activation_function(z)
        return a

    def backpropogate(self, expected_digit, activations):
        # feed forward
        a = [activations]
        z = []
        for b, w in zip(self.biases, self.weights):
            z.append(np.matmul(w, a[-1]) + b)
            a.append(self.calculate_activation_function(z[-1]))

        # initialise weight_changes and bias_changes
        weight_changes = self.create_weight_changes()
        bias_changes = self.create_bias_changes()

        # backpropogate

        # activation_changes[l] is ∂c/∂a^l for the desired layer l
        expected_final_layer = calculate_expected_last_layer(expected_digit)
        activation_changes = self.cost.calculate_cost_derivative(expected_final_layer, a[NUM_LAYERS - 1])

        for l in range(1, NUM_LAYERS)[::-1]:

            dc_by_dz = activation_changes * self.calculate_da_by_dz(a[l])

            weight_changes[l - 1] = np.outer(dc_by_dz, self.calculate_dz_by_dw(a[l - 1]))

            # calculate change to bias needed and add to requested change
            bias_changes[l - 1] = dc_by_dz * self.calculate_dz_by_db()

            # calculate change to activations needed
            activation_changes = np.matmul(dc_by_dz, self.calculate_dz_by_da(self.weights[l - 1]))

        return weight_changes, bias_changes

    def get_normal_cost_term(self, images, labels):
        # return average cost of given images
        cost = 0

        for i in range(len(images)):
            cost += self.cost.calculate_cost(labels[i], self.feed_forward(images[i]))
        cost = cost / len(images)
        return cost

    def get_regularisation_cost_term(self, data_size):
        # l2 regularisation term
        # use data_size as either training or testing data could be given
        weight_squared_sum = sum(np.sum(layer ** 2) for layer in self.weights)
        cost = 0.5 * (REGULARISATION_PARAMETER / data_size) * weight_squared_sum
        return cost

    def get_test_cost(self, images, labels):
        return self.get_normal_cost_term(images, labels) + self.get_regularisation_cost_term(len(images))

    @staticmethod
    def calculate_activation_function(weighted_input):
        # watch out for possible overflows for numbers > +- pow(2, 8)
        activation = 1 / (1 + np.exp(-weighted_input))
        return activation

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
    def calculate_da_by_dz(activation):
        # ∂a/∂z
        # derivative of sigmoid
        return activation * (1 - activation)

    @staticmethod
    def calculate_dz_by_dw(input_activation):
        # ∂z/∂w
        return input_activation

    @staticmethod
    def calculate_dz_by_db():
        # ∂z/∂b
        return 1

    @staticmethod
    def calculate_dz_by_da(weight):
        return weight


def calculate_expected_last_layer(label):
    # returns the expected final layer in network, given the label
    expected_last_layer = np.zeros(NEURONS_IN_LAYER[NUM_LAYERS - 1])
    expected_last_layer[label] = 1.0
    return expected_last_layer
