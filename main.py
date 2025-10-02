from Constants import *
from Image import Image
from NeuralNetwork2 import NeuralNetwork2

nn = NeuralNetwork2()
im = Image()

train_images, train_labels = im.get_training_data(TRAINING_DATA_SIZE)
test_images, test_labels = im.get_testing_data(TESTING_DATA_SIZE)

nn.train_and_show_results_no_improvement_in_n(train_images, train_labels, test_images, test_labels)
