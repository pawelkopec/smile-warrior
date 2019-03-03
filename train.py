import math
import argparse
from data_server import load_dataset
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard


def prepare_data(dataset):
    """
    This module prepares data for training and validation process.

    :param dataset:
        Dataset file path.

    :return:
        6 numpy arrays containing train, test and validate datasets for x and y.
        side_length containing height/width of processed images.
    """
    # Downloading and reshaping data
    x_train, y_train, x_test, y_test, x_validate, y_validate = load_dataset(dataset)
    side_length = int(math.sqrt(x_train.shape[1]))
    x_train = x_train.reshape(x_train.shape[0], side_length, side_length, 1)
    x_test = x_test.reshape(x_test.shape[0], side_length, side_length, 1)
    x_validate = x_validate.reshape(x_validate.shape[0], side_length, side_length, 1)

    x_train, x_test, x_validate = normalize(x_train, x_test, x_validate)

    # Preparing a proper format of output data
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    y_validate = np_utils.to_categorical(y_validate, 2)

    return x_train, y_train, x_test, y_test, x_validate, y_validate, side_length


def normalize(x_train, x_test, x_validate):
    """
    This module performs normalization of input images

    :param x_train:
        Input training data.

    :param x_test:
        Input testing data.

    :param x_validate:
        Input validation data.

    :return:
        Normalized input images.
    """
    return (x_train - 256 / 2) / 256, (x_test - 256 / 2) / 256, (x_validate - 256 / 2) / 256


def prepare_network(shape, learning_rate, saving_period, models_dir, logs_dir):
    """
    This module prepares convolutional neural network model for future training.

    :param shape:
        Height/width of processed images.

    :param learning_rate:
        Length of single step in gradient propagation.

    :param saving_period:
        How many epochs of training must pass before saving model.

    :param models_dir:
        Models saving directory.

    :param logs_dir:
        Logs saving directory.

    :return:
        Randomly initialised model prepared for training.
        Checkpointer object containing parameters needed in model saving.
        Tensorboard object containing parameters needed in logs saving.
    """
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(shape, shape, 1)))

    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=models_dir,
                                   verbose=1, save_best_only=False, save_weights_only=False, period=saving_period)

    tensorboard = TensorBoard(log_dir=logs_dir.format(time()))

    return model, checkpointer, tensorboard


def parse_args():
    """
    This module parses command line arguments.

    :return:
        Parser containing command line data.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', help='input file containing dataset required in training',
                        default='smile_warrior_dataset.csv')

    parser.add_argument('--batch_size', '-b', help='number of images in one batch', type=int, default=32)

    parser.add_argument('--epochs', '-e', help='number of training epochs', type=int, default=10)

    parser.add_argument('--saving_period', '-s', help='how many epochs must pass before saving', type=int, default=1)

    parser.add_argument('--learning_rate', '-lr', help='specifies learning rate', type=float, default=0.0001)

    parser.add_argument('--models_dir', '-md', help='specifies saving directory for models',
                        default='model.{epoch:02d}.hdf5')

    parser.add_argument('--logs_dir', '-ld', help='specifies saving directory for logs', default='logs')

    parser.add_argument('--model_load', '-ml', help='specifies path for loading models', default="model.0{}.hdf5")

    return parser.parse_args()


def main():
    args = parse_args()

    # Network parameters
    batch_size = args.batch_size
    epochs = args.epochs
    saving_period = args.saving_period
    learning_rate = args.learning_rate
    x_train, y_train, x_test, y_test, x_validate, y_validate, side_length = prepare_data(args.dataset)
    model, checkpointer, tensorboard = prepare_network(side_length, learning_rate, saving_period,
                                                       args.models_dir, args.logs_dir)
    model.fit(x_train, y_train,
              batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False,
              validation_data=(x_validate, y_validate), callbacks=[checkpointer, tensorboard])


if __name__ == "__main__":
    main()
