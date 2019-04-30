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

# x_train = np.loadtxt("train_x.txt")
# y_train = np.loadtxt("train_y.txt")
# x_test = np.loadtxt("test_x.txt")
# y_test = np.loadtxt("test_y.txt")


# clf1 = BernoulliNB()
# clf1 = LabelSpreading(kernel="knn")
# clf2 = LabelSpreading(kernel='knn')
# clf2 = BernoulliNB()
# clf1 = MLPClassifier(solver='sgd', activation='logistic')
# clf2 = MLPClassifier(solver='sgd', activation='logistic')
# clf1 = SVC(probability=True)
# clf2 = SVC(probability=True)

data = load_data('covtype.data')
x_train = data['train_x']
y_train = data['train_y']
# validation_x = data['validation_x']
# validation_y = data['validation_y']
x_test = data['test_x']
y_test = data['test_y']

onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train.reshape(len(y_train), 1))
y_test = onehot_encoder.fit_transform(y_test.reshape(len(y_test), 1))



igs = list(np.load('information_gain.npy'))
igs_sorted = sorted(enumerate(igs), key=lambda a: a[1], reverse=True)
igs_sorted_idx = [i for i,ig in igs_sorted]
feature1 = np.array(igs_sorted_idx[::2])
feature2 = np.array(igs_sorted_idx[1::2])
# feature1 = np.arange(0,5)
# feature2 = np.arange(5, 10)
# feature1 = np.append(feature1, np.arange(10, 32))
# feature2 = np.append(feature2, np.arange(32, 54))


all_val_auc = []
all_test_auc = []

for i in range(2,3):
	clf1 = Net(27)
	clf2 = Net(27)

	coTrain = CoTrainingClassifier(clf1, clf2, n=40, p=40, u=300, k=60)

	cur_supervised_num = int(i * 0.1 * 11340)
	indices = np.random.permutation(x_train.shape[0])
	indices = indices.astype(int)

	print("cursupervisednum:", cur_supervised_num)
	# print(x_train.shape)
	
	# print(x_train)
	cur_x_supervised_idx, cur_y_supervised_idx = indices[:cur_supervised_num], indices[:cur_supervised_num]
	# print(cur_x_supervised_idx.shape)
	# print(x_train[np.arange(0,1134),:])
	# print(x_train[cur_x_supervised_idx])
	cur_x_supervised, cur_y_supervised = x_train[cur_x_supervised_idx], y_train[cur_y_supervised_idx]
	cur_x_unsupervised_idx, cur_y_unsupervised_idx = indices[cur_supervised_num:], indices[cur_supervised_num:]
	cur_x_unsupervised, cur_y_unsupervised = x_train[cur_x_unsupervised_idx], y_train[cur_y_unsupervised_idx]

	validation_x = cur_x_supervised[:300]
	validation_y = cur_y_supervised[:300]
	cur_x_supervised = cur_x_supervised[300:]
	cur_y_supervised = cur_y_supervised[300:]
	# feature2 = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
	# feature1 = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
	# features = np.random.permutation(12)
	# s = random.randint(1,12)
	# feature1 = features[:s]
	# feature2 = features[s:]
	# feature1 = np.array([0])
	# feature2 = np.array([1,2,3,6,7,8])
	print("feature1: ", feature1)
	print("feature2: ", feature2)
	cur_x_fit = np.append(cur_x_supervised, cur_x_unsupervised, axis=0)
	cur_y_fit = np.append(cur_y_supervised, np.full(cur_y_unsupervised.shape, 0.0), axis=0)
	# cur_y_fit = np.append(cur_y_supervised, np.full(cur_y_unsupervised.shape, -1.0), axis=0).astype(str)
	assert cur_x_fit.shape[0] == cur_y_fit.shape[0]
	# coTrain.fit(cur_x_fit[:,feature1], cur_x_fit[:, feature2], cur_y_supervised, 
	# 	validation_x[:, feature1], validation_x[:, feature2], validation_y)
	coTrain.fit(cur_x_fit[:,feature1], cur_x_fit[:, feature2], cur_y_fit, 
		validation_x[:, feature1], validation_x[:, feature2], validation_y, i)


	print("START TESTING")
	# onehot_encoder = OneHotEncoder(sparse=False)
	# y_test = onehot_encoder.fit_transform(y_test.reshape(len(y_test), 1))
	preds_val = coTrain.predict_proba(validation_x[:, feature1], validation_x[:, feature2])
	val_auc = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val, axis = 1))
	print("val_auc", val_auc)

	# preds_val_clf1 = coTrain.clf1_.predict_proba(validation_x[:, feature1])
	# print(validation_y.shape, preds_val_clf1.shape)
	# val_auc_clf1 = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val_clf1, axis = 1))
	# print("val_auc_clf1", val_auc_clf1)

	# preds_val_clf2= coTrain.clf2_.predict_proba(validation_x[:, feature2])
	# val_auc_clf2 = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val_clf2, axis = 1))
	# print("val_auc_clf2", val_auc_clf2)

	# print("clf1 clf2 equal %", np.sum(np.equal(np.argmax(preds_val_clf1, axis=1), np.argmax(preds_val_clf2, axis=1))) / 300.0)
	# print("clf1 coTrain equal %", np.sum(np.equal(np.argmax(preds_val_clf1, axis=1), np.argmax(preds_val, axis=1))) / 300.0)
	# print("clf2 coTrain equal %", np.sum(np.equal(np.argmax(preds_val_clf2, axis=1), np.argmax(preds_val, axis=1))) / 300.0)

	preds_test = coTrain.predict_proba(x_test[:, feature1], x_test[:, feature2])
	test_auc = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(preds_test, axis = 1))
	print("test_auc", test_auc)

	preds_test_clf1= coTrain.clf1_.predict_proba(x_test[:, feature1])
	# test_auc_clf1 = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(preds_test_clf1, axis = 1))
	# print("test_auc_clf1", test_auc_clf1)

	# preds_test_clf2= coTrain.clf2_.predict_proba(x_test[:, feature2])
	# test_auc_clf2 = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(preds_test_clf2, axis = 1))
	# print("test_auc_clf2", test_auc_clf2)

	all_val_auc.append(val_auc)
	all_test_auc.append(test_auc)


	# test_error = coTrain.test_error(x_test[:10000, feature1], x_test[:10000, feature2], y_test[:10000])
	# print("test error: ", test_error)


print('all_val_auc: ', all_val_auc)
print('all_test_auc: ', all_test_auc)



