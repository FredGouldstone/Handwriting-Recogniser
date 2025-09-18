from Constants import *
from Image import Image
from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork()
im = Image()

train_images, train_labels = im.get_training_data(TRAINING_DATA_SIZE)
test_images, test_labels = im.get_testing_data(TESTING_DATA_SIZE)

nn.train_with_progress_indicators(train_images, train_labels, test_images, test_labels)