from matplotlib import pyplot as plt
import numpy as np
from Constants import *


class Image:
    def __init__(self):
        self.train_images, self.train_labels = self.load_images(TRAINING_DATA_FILEPATH, TRAINING_DATA_SIZE)
        self.test_images, self.test_labels = self.load_images(TESTING_DATA_FILEPATH, TESTING_DATA_SIZE)

    def load_images(self, filepath, data_size):
        # each line is label + image
        # label is first character
        # image stored as csv
        images = []
        labels = []

        file = open(filepath, mode='r')
        file.readline()  # first line of data shows layout

        for image_index in range(data_size):
            line = file.readline().strip()  # remove newline
            label_and_pixels = line.split(",")

            label = int(label_and_pixels[0])

            # the rest is pixels
            # divide by 255 as should be between 0 and 1 for NN
            pixels = np.array(label_and_pixels[1:], dtype=np.float32)
            pixels = pixels - 1
            pixels = np.clip(pixels, 0.0, 255.0)
            pixels = pixels / 255.0

            images.append(pixels)
            labels.append(label)

        file.close()

        return images, labels

    def display_image(self, image_array, label):
        # takes 1D array of activations between 0 and 1 and shows image

        # convert to between 0 and 255
        image_array = image_array * 255

        # reshape
        image_array = np.reshape(image_array, IMAGE_DIMENSIONS)

        # show image
        title = 'label = ' + str(label)
        plt.title(title)
        plt.imshow(image_array, interpolation='nearest')
        plt.show()

    def get_train_image(self, image_number):
        return self.train_images[image_number]

    def get_test_image(self, image_number):
        return self.test_images[image_number]

    def get_train_label(self, image_number):
        return self.train_labels[image_number]

    def get_test_label(self, image_number):
        return self.test_labels[image_number]

    def get_training_data(self, length):
        images = self.train_images[0:length]
        labels = self.train_labels[0:length]
        return images, labels

    def get_testing_data(self, length):
        images = self.test_images[0:length]
        labels = self.test_labels[0:length]
        return images, labels
