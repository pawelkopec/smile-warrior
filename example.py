# Example how to load dataset by function from module and to show single picture

import sys

# You may need to execute line below:
# sys.path.append("Your path to project file for example: B:/Gradient/Smile_warrior")

from data_server import *

# You have to firstly create smile_warrior_dataset by running script 'Adjust_data_set' which creates that data_set from
# fer2013 data set
filename = 'smile_warrior_dataset.csv'

X_train, Y_train, X_test, Y_test, X_validate, Y_validate = load_dataset(filename)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_validate.shape, Y_validate.shape)

show_picture(9, X_train)




