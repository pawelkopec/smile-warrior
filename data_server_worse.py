#Nothing interesting, script in need of improvement

import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt


data_set = pd.read_csv("fer2013.csv")


# Preview the first 20 lines of the loaded data
print("Before:")
print(data_set.head(n=20))

print(data_set.shape)

training_sample = 0
test_sample = 0
validation_sample= 0

for index,row in data_set.iterrows():
    if (row['emotion'] == 3):
        data_set.at[index, 'emotion'] = 1
    elif (row['emotion'] != 3):
        data_set.at[index, 'emotion'] = 0

    if (row['Usage'] == 'Training'):
        training_sample= training_sample + 1
    elif (row['Usage'] == 'PublicTest'):
        test_sample= test_sample + 1
    elif (row['Usage'] == 'PrivateTest'):
        validation_sample= validation_sample + 1


print(training_sample)      #28709
print(test_sample)          #3589
print(validation_sample)    #3589

print("After:")
print(data_set.head(n=20))

X_data = np.empty((0, 2304))
Y_data = np.array([])


print("Creating data_set ...")

for _, row in data_set.iterrows():
    numpy_row = np.array([])
    numpy_row = np.append(numpy_row, [float(s) for s in row['pixels'].split()])
    X_data = np.append(X_data, [numpy_row], axis=0)
    Y_data = np.append(Y_data, float(row['emotion']))


print("X_data i Y_data shape")
print(X_data.shape)
print(Y_data.shape)

np.savetxt("X_data.csv", X_data, delimiter=",")
np.savetxt("Y_data.csv", Y_data, delimiter=",")


