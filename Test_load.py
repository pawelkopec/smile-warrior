#Test script to load data by function

import sys
sys.path.append("Your path to project file")
from data_server import *

filename = 'smile_warrior_dataset.csv'

X_train, Y_train, X_test, Y_test, X_validate, Y_validate = load_data(filename)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_validate.shape, Y_validate.shape)

Show_Picture(3, X_train)




