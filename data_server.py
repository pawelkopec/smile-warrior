#Nothing interesting, script in need of improvement

import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt

"""
my_data = genfromtxt('fer2013.csv', delimiter=',', skip_header =1)

print(my_data.shape)

print(my_data[0])
print(my_data[1])
print(my_data[2])

print("Przed zamianą")

for i in range(20):
    print("Probka nr %d: %d" % (i+1, my_data[i, 0]))


for i in range(my_data.shape[0]):
    if(my_data[i,0] == 3):
        my_data[i, 0] = 1
    elif (my_data[i,0] != 3):
        my_data[i, 0] = 0


print("Po zamianie")

for i in range(20):
    print("Probka nr %d: %d" % (i+1, my_data[i, 0]))

print(my_data.shape)

print(my_data[0])
print(my_data[1])
print(my_data[2])

np.savetxt("smile_warrior_dataset.csv", my_data, delimiter=",")
"""
data_set = pd.read_csv("fer2013.csv")
# Preview the first 5 lines of the loaded data

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

#data_set.to_csv('smile_warrior_dataset.csv')

numpy_test = np.array([])
for i in range(5):
    numpy_test = np.append(numpy_test, i)

print(numpy_test)
print(type(numpy_test))
print(numpy_test.shape)

numpy_test2 = np.array([])
numpy_test2 = np.append(numpy_test2, [i for i in range(5)])
print(numpy_test2)
print(type(numpy_test2))
print(numpy_test2.shape)

print("Creating data_set ...")

for index,row in data_set.iterrows():
    numpy_row = np.array([])
    #for s in row['pixels'].split():
        #pixel_numpy[index, counter] = float(s)
        #numpy_row = np.append(numpy_row, float(s))
    numpy_row = np.append(numpy_row, [float(s) for s in row['pixels'].split()])
    #print(numpy_row.shape)
    X_data = np.append(X_data, [numpy_row], axis=0)
    Y_data = np.append(Y_data, float(row['emotion']))
    #np.append(pixel_numpy, X)
    #X.append(pixel_numpy)

#print(pixel_array)
#pixel_numpy = np.array(pixel_array)
"""
print(pixel_numpy.shape)
print(pixel_numpy[0,0:10])
#print(pixel_numpy)
np.savetxt("temp.csv", pixel_numpy, delimiter=",")
pixel_numpy = np.reshape(pixel_numpy, (3, 48,48))
print(pixel_numpy.shape)
plt.imshow(pixel_numpy[0])
plt.show()
"""
# a = np.array([[1],[2],[3]])
# b = np.array([[1],[2],[3]])
#
# print(a)
# print(b)
#
# c = np.column_stack((a,b))
# print(c)
#
# d = np.array([1,2,3])
# print(d)
# print(d.shape)
#
# e = np.array([4,5,6])
# print(e)
#
# f = np.array([])
# print(f)
#
# f = np.concatenate((f,d.T),axis=0)
# print(f)
# f = np.concatenate((f,e.T),axis=0)
# print(f)
# print(f.shape)
# g = np.copy(e)
# g = np.vstack((g,d))
# print(g.shape)
# print(g)
#
# print(X_data.shape)
#
#
# arr = np.empty((0,3), int)
#
# arr = np.append(arr, np.array([[1,2,3]]), axis=0)
# arr = np.append(arr, np.array([[4,5,6]]), axis=0)
#
# print(arr)
# print(arr.shape)

print("X_data i Y_data shape")
print(X_data.shape)
print(Y_data.shape)

np.savetxt("X_data.csv", X_data, delimiter=",")
np.savetxt("Y_data.csv", Y_data, delimiter=",")




#print(pd.to_numeric(row['pixels']))





