from EM_sol import load_data, NaiveBayesSemiSupervised
import numpy as np
import sys


data_set = load_data()

x_supervised = np.loadtxt("train_x.txt")
y_supervised = np.loadtxt("train_y.txt")
x_unsupervised = np.loadtxt("validation_x.txt")
x_test = np.loadtxt("test_x.txt")
y_test = np.loadtxt("test_y.txt")

print(np.unique(y_supervised))


print(np.shape(x_supervised), flush=True)
print(x_supervised, flush=True)

data_set = dict()
EM = NaiveBayesSemiSupervised(data_set)
EM.train(x_supervised, x_unsupervised, y_supervised)

print("test_error: ", EM.test_error(x_test, y_test))

print("train_error: ", EM.test_error(x_supervised, y_supervised))
