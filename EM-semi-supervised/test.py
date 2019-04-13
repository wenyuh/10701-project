from EM_sol import load_data, NaiveBayesSemiSupervised
import numpy as np
import sys

# Load the input data from the .txt files and push it through the 
# semi-supervised EM pipeline using Naive Bayes 
#

# data_set = load_data()

# np.savetxt("train_x.txt", data_set['train_x'])
# np.savetxt("train_y.txt", data_set['train_y'])
# np.savetxt("test_x.txt", data_set['test_x'])
# np.savetxt("test_y.txt", data_set['test_y'])


x_train = np.loadtxt("train_x.txt")
y_train = np.loadtxt("train_y.txt")
# x_unsupervised = np.loadtxt("validation_x.txt")
x_test = np.loadtxt("test_x.txt")
y_test = np.loadtxt("test_y.txt")

# print(np.unique(y_supervised))


# print(np.shape(x_supervised), flush=True)
# print(x_supervised, flush=True)


for i in range(1, 6): # 10%, 20%, ..., 50% supervised data
	print("i = ", i)
	data_set = dict()
	EM = NaiveBayesSemiSupervised(data_set)
	cur_supervised_num = int(i * 0.1 * 11340)
	indices = np.random.permutation(x_train.shape[0])
	indices = indices.astype(int)

	print(cur_supervised_num)
	# print(x_train.shape)
	
	# print(x_train)
	cur_x_supervised_idx, cur_y_supervised_idx = indices[:cur_supervised_num], indices[:cur_supervised_num]
	# print(cur_x_supervised_idx.shape)
	# print(x_train[np.arange(0,1134),:])
	# print(x_train[cur_x_supervised_idx])
	cur_x_supervised, cur_y_supervised = x_train[cur_x_supervised_idx], y_train[cur_y_supervised_idx]
	cur_x_unsupervised_idx, cur_y_unsupervised_idx = indices[cur_supervised_num:], indices[cur_supervised_num:]
	cur_x_unsupervised, cur_y_unsupervised = x_train[cur_x_unsupervised_idx], y_train[cur_y_unsupervised_idx]

	EM.train(cur_x_supervised, cur_x_unsupervised, cur_y_supervised)

	print("test_error: ", EM.test_error(x_test, y_test))

	print("train_error: ", EM.test_error(cur_x_supervised, cur_y_supervised))



# data_set = dict()
# EM = NaiveBayesSemiSupervised(data_set)
# EM.train(x_supervised, x_unsupervised, y_supervised)

# print("test_error: ", EM.test_error(x_test, y_test))

# print("train_error: ", EM.test_error(x_supervised, y_supervised))
