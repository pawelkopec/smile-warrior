import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
import sys
from keras.callbacks import ModelCheckpoint
sys.path.append('\PycharmProjects\smile-warrior')
from data_server import *


#Downloading and reshaping data
X_train, Y_train, X_test, Y_test, X_validate, Y_validate = Load_dataset('smile_warrior_dataset.csv')
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
X_validate = X_validate.reshape(X_validate.shape[0], 48, 48, 1)


#Normalization
mean_image = np.mean(X_train)
std_image = np.std(X_train)
X_train = (X_train - mean_image) / std_image
X_test = (X_test - mean_image) / std_image
X_validate = (X_validate - mean_image) / std_image


#Preparing a proper format of output data
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)
Y_validate = np_utils.to_categorical(Y_validate, 2)


#Network parameters
Batch_size = 32
Epochs = 5
Saving_period = 1
Learning_rate = 0.001


#Preparing neural network model
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation = 'relu', input_shape = (48,48,1)))
model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation ='softmax'))
opt = optimizers.Adam(lr = Learning_rate)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
checkpointer = ModelCheckpoint(filepath = 'D:\Python3.6.6\Jupyter\smile_warrior\weights.{epoch:02d}.hdf5',
                               verbose = 1, save_best_only = False, period = Saving_period)

#Saving model
model.save('model.hdf5')


#Training
model.fit(X_train, Y_train,
          batch_size = Batch_size, epochs = Epochs, verbose = 1, shuffle = True,
          validation_data = (X_validate, Y_validate), callbacks = [checkpointer])


#Downloading model
#model = load_model('model.hdf5')


for i in range((int)(Epochs/Saving_period)):
    #Downloading weights
    model.load_weights("D:\Python3.6.6\Jupyter\smile_warrior\weights.0{}.hdf5".format(i+1))
    #Checking results with testing dataset
    score_test = model.evaluate(X_test, Y_test, verbose = 1)
    print("Score after {}. epoch: ".format(i+1))
    print(score_test)
