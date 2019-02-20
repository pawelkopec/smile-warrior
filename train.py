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
model.save('model1.f5')


#Downloading model
model = load_model('model.hdf5')


