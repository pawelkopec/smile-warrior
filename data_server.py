# Module to load dataset and divide it into train, test and validation sets for X and Y

import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):

    with open(filename) as file:

        next(file)  # Skipping first line of file which contains name of data columns

        X_list_train = []
        X_list_test = []
        X_list_valid = []

        Y_list_train = []
        Y_list_test = []
        Y_list_valid = []

        print("Preparing dataset...")

        for line in file:

            row = line.split(',')

            # Pixel values for each picture are saved in csv in single cells as a string containing values splited
            # by space-bar so we need to split that string to substrings, convert them to ints and add these pixel
            # values to list one by one

            if row[2] == "Training\n":
                Y_list_train.append(int(row[0]))
                X_list_train.append([int(pixel) for pixel in row[1].split()])

            elif row[2] == "PublicTest\n":
                Y_list_test.append(int(row[0]))
                X_list_test.append([int(pixel) for pixel in row[1].split()])

            elif row[2] == "PrivateTest\n":
                Y_list_valid.append(int(row[0]))
                X_list_valid.append([int(pixel) for pixel in row[1].split()])

        X_train = np.array(X_list_train)
        X_test = np.array(X_list_test)
        X_validate = np.array(X_list_valid)

        Y_train = np.array(Y_list_train)
        Y_test = np.array(Y_list_test)
        Y_validate = np.array(Y_list_valid)

        print("Dataset prepared")

        return X_train, Y_train, X_test, Y_test, X_validate, Y_validate


# Showing single picture from X_data numpy array, number of picture is controled by variable 'which_one'
def show_picture(which_one, x_data):

    plt.imshow(x_data[which_one, :].reshape(48, 48))
    plt.show()
