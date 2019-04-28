import numpy as np
import sys
from cotrain import CoTrainingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.semi_supervised import LabelSpreading
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from nn_class import Net, load_data
import random



data = load_data('covtype.data')
x_train = data['train_x']
y_train = data['train_y']
# validation_x = data['validation_x']
# validation_y = data['validation_y']
x_test = data['test_x']
y_test = data['test_y']

# np.savetxt("train_x.txt", x_train)
# np.savetxt("train_y.txt", y_train)
# np.savetxt("test_x.txt", x_train)
# np.savetxt("test_y.txt", y_train)


# x_train = np.loadtxt("train_x.txt")
# y_train = np.loadtxt("train_y.txt")
# x_test = np.loadtxt("test_x.txt")
# y_test = np.loadtxt("test_y.txt")

onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train.reshape(len(y_train), 1))
y_test = onehot_encoder.fit_transform(y_test.reshape(len(y_test), 1))


igs = list(np.load('information_gain.npy'))
igs_sorted = sorted(enumerate(igs), key=lambda a: a[1], reverse=True)
igs_sorted_idx = [i for i,ig in igs_sorted]

feature1 = np.arange(0,5)
feature2 = np.arange(5, 10)
feature1 = np.append(feature1, np.arange(10, 32))
feature2 = np.append(feature2, np.arange(32, 54))


NN1 = Net(27)
# NN1 = Net(54)
NN2 = Net(27)


	


def test():
	cur_supervised_num = int(2 * 0.1 * 11340)
	indices = np.random.permutation(x_train.shape[0])
	indices = indices.astype(int)

	# print(cur_supervised_num)
	# print(x_train.shape)
	
	# print(x_train)
	cur_x_supervised_idx, cur_y_supervised_idx = indices[:cur_supervised_num], indices[:cur_supervised_num]
	# print(cur_x_supervised_idx.shape)
	# print(x_train[np.arange(0,1134),:])
	# print(x_train[cur_x_supervised_idx])
	cur_x_supervised, cur_y_supervised = x_train[cur_x_supervised_idx], y_train[cur_y_supervised_idx]
	cur_x_unsupervised_idx, cur_y_unsupervised_idx = indices[cur_supervised_num:], indices[cur_supervised_num:]
	cur_x_unsupervised, cur_y_unsupervised = x_train[cur_x_unsupervised_idx], y_train[cur_y_unsupervised_idx]

	validation_x = cur_x_supervised[:300, :]
	validation_y = cur_y_supervised[:300, :]
	# validation_y = onehot_encoder.fit_transform(validation_y.reshape(len(validation_y), 1))

	cur_x_supervised = cur_x_supervised[300:, :]
	cur_y_supervised = cur_y_supervised[300:, :]

	NN1.fit(cur_x_supervised[:, feature1], cur_y_supervised, 
		validation_x[:, feature1], validation_y)

	# NN1.fit(cur_x_supervised, cur_y_supervised, 
	# 	validation_x, validation_y)

	NN2.fit(cur_x_supervised[:, feature2], cur_y_supervised, 
		validation_x[:, feature2], validation_y)






	print("NN1 test")
	model = NN1.model
	preds_val = model.predict(validation_x[:, feature1])
	val_auc = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val, axis = 1))

	print("val_auc", val_auc)

	preds_test = model.predict(x_test[:, feature1], verbose=1)
	test_auc = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(preds_test, axis = 1))
    
	print("test_auc", test_auc)

	# print("NN1 test")
	# model = NN1.model
	# preds_val = model.predict(validation_x)
	# val_auc = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val, axis = 1))

	# print("val_auc", val_auc)

	# preds_test = model.predict(x_test, verbose=1)
	# test_auc = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(preds_test, axis = 1))
    
	# print("test_auc", test_auc)


	print("NN2 test")
	model2 = NN2.model
	preds_val2 = model2.predict(validation_x[:, feature2])
	val_auc2 = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val, axis = 1))

	print("val_auc2", val_auc2)

	preds_test2 = model2.predict(x_test[:, feature2], verbose=1)
	test_auc2 = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(preds_test2, axis = 1))
    
	print("test_auc2", test_auc2)


test()
