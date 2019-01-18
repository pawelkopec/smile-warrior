#Module to load data set and divide it into train, test and validation sets

def load_data(filename):

    import numpy as np

    try:
        file = open(filename)
    except:
        print("Unable to open the file")
    else:
        print("File opened")

    X_list = []
    Y_list = []

    first_line = True

    print("Converting data...")

    for line in file:
        if first_line == True:
            first_line = False
        else:
            row = line.split(',')
            Y_list.append(int(row[0]))
            X_list.append([int(pixel) for pixel in row[1].split()])

    X_data = np.array(X_list)
    Y_data = np.array(Y_list)

    print("Data converted")

    print("X_data shape: %d %d" % (X_data.shape))
    print("Y_data shape: %d " % (Y_data.shape))

    file.close()

    training_sample = 28709
    test_sample = 3589
    validation_sample = 3589

    X_train = X_data[0:training_sample, :]
    X_test = X_data[training_sample:training_sample + test_sample, :]
    X_validate = X_data[training_sample + test_sample:training_sample + test_sample + validation_sample, :]

    Y_train = Y_data[0:training_sample]
    Y_validate = Y_data[training_sample:training_sample + test_sample]
    Y_test = Y_data[training_sample + test_sample:training_sample + test_sample + validation_sample]

    return X_train, Y_train, X_test, Y_test, X_validate, Y_validate
