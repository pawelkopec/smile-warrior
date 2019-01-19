#Script changing labels of emotions to 1 - smile, 0 - no smile and saving changed dataset to new csv file

import pandas as pd

data_set = pd.read_csv("fer2013.csv")

print("Before:")
print(data_set.head(n=20))

print(data_set.shape)

#Counting size of train, test and validation sets
training_sample = 0
test_sample = 0
validation_sample= 0

print("Adjusting dataset... ")

#Emotion with label 3 is happines to we change label to 1 for our smile detector, other emotions are changed to 0.
for index,row in data_set.iterrows():
    if (row['emotion'] == 3):
        data_set.at[index, 'emotion'] = 1
    else:
        data_set.at[index, 'emotion'] = 0

    if (row['Usage'] == 'Training'):
        training_sample= training_sample + 1
    elif (row['Usage'] == 'PublicTest'):
        test_sample= test_sample + 1
    elif (row['Usage'] == 'PrivateTest'):
        validation_sample= validation_sample + 1


print("Size of training data set: %d " % training_sample)         #28709
print("Size of test data set: %d " % test_sample)                 #3589
print("Size of validation data set: %d " % validation_sample)     #3589

print("After:")
print(data_set.head(n=20))

data_set.to_csv('smile_warrior_dataset.csv', index=False)
print("Smile-Warrior dataset created in file 'smile_warrior_dataset.csv'")

