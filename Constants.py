# filepaths
TRAINING_DATA_FILEPATH = '/Users/fredgouldstone/PycharmProjects/PhotoNeuralNetwork/mnist-in-csv/mnist_train.csv'
TESTING_DATA_FILEPATH = '/Users/fredgouldstone/PycharmProjects/PhotoNeuralNetwork/mnist-in-csv/mnist_test.csv'

# NN parameters
IMAGE_DIMENSIONS = [28, 28]
NEURONS_IN_LAYER = [IMAGE_DIMENSIONS[0] * IMAGE_DIMENSIONS[1], 30, 10]
NUM_LAYERS = len(NEURONS_IN_LAYER)
BATCH_SIZE = 10
LEARNING_RATE = 3
EPOCHS = 30
TRAINING_DATA_SIZE = 50000

# testing parameters
TESTING_DATA_SIZE = 10000
