#Module to load data set and divide it into train, test and validation sets for X and Y

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

    print("Preparing data...")

    for line in file:
        if first_line == True:
            first_line = False
        else:
            row = line.split(',')
            Y_list.append(int(row[0]))
            X_list.append([int(pixel) for pixel in row[1].split()])

    X_data = np.array(X_list)
    Y_data = np.array(Y_list)

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

    print("Data prepared")

    return X_train, Y_train, X_test, Y_test, X_validate, Y_validate



def Show_Picture(wchich_one, X_data):

    import matplotlib.pyplot as plt

    plt.imshow(X_data[wchich_one,:].reshape(48, 48))
    plt.show()

    return None