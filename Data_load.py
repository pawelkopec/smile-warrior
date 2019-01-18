#Nothing interesting, script in need of improvement

from numpy import genfromtxt

training_sample = 28709
test_sample = 3589
validation_sample= 3589

X = genfromtxt('X_data.csv', delimiter=',')
Y = genfromtxt('Y_data.csv', delimiter=',')

print("Wczytane")

X_train = X[0:training_sample,:]
X_validate = X[training_sample:training_sample+test_sample, :]
X_test = X[training_sample+test_sample:training_sample+test_sample+validation_sample, :]

Y_train = Y[0:training_sample,:]
Y_validate = Y[training_sample:training_sample+test_sample, :]
Y_test = Y[training_sample+test_sample:training_sample+test_sample+validation_sample, :]

print("Shape:")

print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)

print(Y_train.shape)
print(Y_validate.shape)
print(Y_test.shape)
