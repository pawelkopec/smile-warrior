# Module to load dataset and divide it into train, test and validation sets for X and Y
import numpy as np
import matplotlib.pyplot as plt
import math

usage = ["Training\n", "PublicTest\n", "PrivateTest\n"]


def load_dataset(filename):
    """
    Function loads dataset from csv file at path passed as param 'filename'.
    Dataset is divided into train, test and validate parts based on values in column 'Usage' in csv file.

    You have to firstly create Smile-Warrior dataset by running script 'adjust_data_set' which creates it
    from fer2013.csv dataset.

    Example:

        x_train, y_train, x_test, y_test, x_validate, y_validate = load_dataset(filename)

    :param filename:
        Path to csv file with dataset.

    :return:
        6 numpy arrays containing train, test and validate datasets for X and Y.
    """

    with open(filename) as file:

        next(file)  # Skipping first line of file which contains name of data columns

        x_list_train = []
        x_list_test = []
        x_list_valid = []

        y_list_train = []
        y_list_test = []
        y_list_valid = []

        print("Preparing dataset...")

        for line in file:

            row = line.split(',')

            # Pixel values for each picture are saved in csv in single cells as a string containing values splited
            # by space-bar so we need to split that string to substrings, convert them to ints and add these pixel
            # values to list one by one

            if row[2] == usage[0]:
                y_list_train.append(int(row[0]))
                x_list_train.append([int(pixel) for pixel in row[1].split()])

            elif row[2] == usage[1]:
                y_list_test.append(int(row[0]))
                x_list_test.append([int(pixel) for pixel in row[1].split()])

            elif row[2] == usage[2]:
                y_list_valid.append(int(row[0]))
                x_list_valid.append([int(pixel) for pixel in row[1].split()])

        x_train = np.array(x_list_train)
        x_test = np.array(x_list_test)
        x_validate = np.array(x_list_valid)

        y_train = np.array(y_list_train)
        y_test = np.array(y_list_test)
        y_validate = np.array(y_list_valid)

        print("Dataset prepared")

        return x_train, y_train, x_test, y_test, x_validate, y_validate


def show_picture(picture, x_dim, y_dim):
    """
    Showing single picture passed as an input param in form of a numpy array.

    Function reshapes input numpy array to shape (x_dim, y_dim) and then uses matplotlib.pyplot.imshow to plot it.

    :param picture:
        Picture in form of numpy array containing consecutive values of pixels, this numpy array should be appropriate
        to change its shape to (x_dim, y_dim).

    :param x_dim:
        Dimension of showed picture in X axis.

    :param y_dim:
        Dimension of showed picture in Y axis.
    """

    plt.imshow(picture.reshape(x_dim, y_dim))
    plt.show()
