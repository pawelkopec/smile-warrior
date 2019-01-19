#Script loading data set (value of emotions and values of pixels) to numpy arrays

import numpy as np

filename = 'smile_warrior_dataset.csv'

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
    if first_line==True:        #Skipping first line of file which contains name of data columns
        first_line = False
    else:
        row = line.split(',')
        Y_list.append(int(row[0]))
        #Pixel values for each picture are saved in csv in single cells as a string containing values splited by
        #space-bar so we need to split that string to substrings, convert them to ints and add these pixel values to list one by one
        X_list.append([int(pixel) for pixel in row[1].split()])

X_data = np.array(X_list)
Y_data = np.array(Y_list)

print("Data converted")

print("X_data shape: %d %d" % (X_data.shape))
print("Y_data shape: %d " % (Y_data.shape))

file.close()