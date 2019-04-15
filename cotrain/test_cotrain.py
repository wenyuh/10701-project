import numpy as np
import sys
from cotrain import CoTrainingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.semi_supervised import LabelSpreading
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from nn_class import Net
import random

x_train = np.loadtxt("train_x.txt")
y_train = np.loadtxt("train_y.txt")
x_test = np.loadtxt("test_x.txt")
y_test = np.loadtxt("test_y.txt")


# clf1 = BernoulliNB()
# clf1 = LabelSpreading(kernel="knn")
# clf2 = LabelSpreading(kernel='knn')
# clf2 = BernoulliNB()
# clf1 = MLPClassifier(solver='sgd', activation='logistic')
# clf2 = MLPClassifier(solver='sgd', activation='logistic')
# clf1 = SVC(probability=True)
# clf2 = SVC(probability=True)
clf1 = Net()
clf2 = Net()

coTrain = CoTrainingClassifier(clf1, clf2, n=5, p=5)

igs = list(np.load('information_gain.npy'))
igs_sorted = sorted([for i,ig in enumerate(igs)], key=lambda a: a[1], reverse=True)
igs_sorted_idx = [i for i,ig in igs_sorted]


for i in range(1):
	cur_supervised_num = int(5 * 0.1 * 11340)
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

	# feature2 = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
	# feature1 = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
	# features = np.random.permutation(12)
	# s = random.randint(1,12)
	# feature1 = features[:s]
	# feature2 = features[s:]
	feature1 = np.array([0])
	feature2 = np.array([1,2,3,6,7,8])
	print("feature1: ", feature1)
	print("feature2: ", feature2)
	cur_x_fit = np.append(cur_x_supervised, cur_x_unsupervised, axis=0)
	cur_y_fit = np.append(cur_y_supervised, np.full(cur_y_unsupervised.shape, -1.0))
	coTrain.fit(cur_x_fit[:,feature1], cur_x_fit[:, feature2], cur_y_fit)

	
	test_error = coTrain.test_error(x_test[:, feature1], x_test[:, feature2], y_test)
	print("test error: ", test_error)






