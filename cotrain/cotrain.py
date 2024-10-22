import numpy as np
import random
import copy
from sklearn.metrics import roc_auc_score, accuracy_score

class CoTrainingClassifier(object):
	"""
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).
	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.
	p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)
	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)
	k - (Optional) The number of iterations
		The default is 30 (from paper)
	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

	def __init__(self, clf, clf2=None, p=-1, n=-1, k=30, u = 75):
		self.clf1_ = clf
		
		#we will just use a copy of clf (the same kind of classifier) if clf2 is not specified
		if clf2 == None:
			self.clf2_ = copy.copy(clf)
		else:
			self.clf2_ = clf2

		#if they only specify one of n or p, through an exception
		if (p == -1 and n != -1) or (p != -1 and n == -1):
			raise ValueError('Current implementation supports either both p and n being specified, or neither')

		self.p_ = p
		self.n_ = n
		self.k_ = k
		self.u_ = u

		random.seed()


	def fit(self, X1, X2, y, validation_x1, validation_x2, validation_y, data_percent):
		"""
		Description:
		fits the classifiers on the partially labeled data, y.
		Parameters:
		X1 - array-like (n_samples, n_features_1): first set of features for samples
		X2 - array-like (n_samples, n_features_2): second set of features for samples
		y - array-like (n_samples): labels for samples, -1 indicates unlabeled
		"""

		#we need y to be a numpy array so we can do more complex slicing
		# y = np.asarray(y)

		#set the n and p parameters if we need to
		# if self.p_ == -1 and self.n_ == -1:
		# 	num_pos = sum(1 for y_i in y if y_i == 1)
		# 	num_neg = sum(1 for y_i in y if y_i == 0)
			
		# 	n_p_ratio = num_neg / float(num_pos)
		
		# 	if n_p_ratio > 1:
		# 		self.p_ = 1
		# 		self.n_ = round(self.p_*n_p_ratio)

		# 	else:
		# 		self.n_ = 1
		# 		self.p_ = round(self.n_/n_p_ratio)

		# assert(self.p_ > 0 and self.n_ > 0 and self.k_ > 0 and self.u_ > 0)

		#the set of unlabeled samples
		U = np.arange(data_percent * 0.1 * 11340 - 300, 11340 - 300).astype(int)
		# U = [i for i, y_i in enumerate(y) if y_i == -1]

		#we randomize here, and then just take from the back so we don't have to sample every time
		random.shuffle(U)

		#this is U' in paper
		U_ = U[-min(len(U), self.u_):].astype(int)

		#the samples that are initially labeled
		# L = [i for i, y_i in enumerate(y) if y_i != -1]
		L = np.arange(0, data_percent * 0.1 * 11340-300).astype(int)

		#remove the samples in U_ from U
		U = U[:-len(U_)]

		self.epoch_train_auc = [0]*(self.k_+1)
		self.epoch_val_auc = [0] * (self.k_ + 1)
		it = 0 #number of cotraining iterations we've done so far

		#loop until we have assigned labels to everything in U or we hit our iteration break condition
		while it != self.k_ and U.shape[0] > self.n_*7:
			it += 1

			assert X1[L].shape[0] == y[L].shape[0]
			assert validation_x1.shape[0] == validation_y.shape[0]
			# print(X1.shape, y.shape, validation_x1.shape, validation_y.shape)
			# print("X1:",X1)
			# print("y", y)
			# print("validation_x1", validation_x1)
			# print("validation_y", validation_y)
			self.clf1_.fit(X1[L], y[L], validation_x1, validation_y)
			self.clf2_.fit(X2[L], y[L], validation_x2, validation_y)

			y1 = self.clf1_.predict(X1[U_])
			y2 = self.clf2_.predict(X2[U_])

			# print("y1: ", y1)

			y1_prob = self.clf1_.predict_proba(X1[U_])
			y2_prob = self.clf2_.predict_proba(X2[U_])

			# print(y1_prob)
			y1_prob_max = [max(lst) for lst in y1_prob]
			y2_prob_max = [max(lst) for lst in y2_prob]

			
			y1_topN1 = zip(y1, y1_prob_max) # (index at U_, index at X/Y, label(col#), prob)
			y1_topN1 = [(i, U_[i], a[0], a[1]) for i,a in enumerate(y1_topN1)]
			y2_topN1 = zip(y2, y2_prob_max)
			y2_topN1 = [(i, U_[i], a[0], a[1]) for i,a in enumerate(y2_topN1)]
			assert len(y1_topN1) == len(U_)

			y1_topN = []
			y2_topN = []
			for k in range(7):
				label_k1 = filter(lambda a: a[2] == k, y1_topN1)
				label_k2 = filter(lambda a: a[2] == k, y2_topN1)
				y1_topN.extend(sorted(label_k1, key=lambda a: a[3], reverse=True)[:self.n_])
				y2_topN.extend(sorted(label_k2, key=lambda a: a[3], reverse=True)[:self.n_])




			# y1_topN = sorted(y1_topN1, key=lambda a: a[3], reverse=True)[:self.n_]
			# y2_topN = sorted(y2_topN1, key=lambda a: a[3], reverse=True)[:self.n_]

			# print("y1_topN: ", y1_topN)
			# print("y2_topN: ", y2_topN)
			self.theta1_ = 0.9
			self.theta2_ = 0.9
			y1_topN_theta = list(filter(lambda a: a[3] > self.theta1_, y1_topN))
			y2_topN_theta = list(filter(lambda a: a[3] > self.theta2_, y2_topN))

			# print("U len: ", len(U_))
			# print("len1, len2 = ", len(y1_topN_theta), len(y2_topN_theta))
			# for k in range(7):
				# print("k=%d: %d, %d",k, len(list(filter(lambda a: a[2] == k, y1_topN_theta))),
				# 	len(list(filter(lambda a: a[2] == k, y2_topN_theta))))

			if (len(y1_topN_theta) > len(y2_topN_theta)):
				# L.extend([a[1] for a in y1_topN])
				# for a in sorted(y1_topN, key=lambda a: a[0], reverse=True):
				# 	y[a[1]] = a[2]
				# 	# print("a[0]: ", a[0])
				# 	U_.pop(a[0])
				# num_to_add = len(y1_topN)

				L = np.append(L, [a[1] for a in y1_topN_theta])
				# L.extend([a[1] for a in y1_topN_theta])
				for a in sorted(y1_topN_theta, key=lambda a: a[0], reverse=True):
					y[a[1]] = np.full(y[0].shape, 0)
					y[a[1]][a[2]] = 1
					# y = y.astype(str)
					# y[a[1]] = a[2]
					# print("a[0]: ", a[0])
					# U_.pop(a[0])
					if (a[0] < U_.shape[0]-1):
						U_ = np.append(U_[:a[0]], U_[a[0]+1:], axis=0)
					else:
						U_ = U_[:a[0]]
				num_to_add = len(y1_topN_theta)

			else:
				# L.extend([a[1] for a in y2_topN])
				# for a in sorted(y2_topN, key=lambda a: a[0], reverse=True):
				# 	y[a[1]] = a[2]		
				# 	U_.pop(a[0])	
				# num_to_add = len(y2_topN)	
				L = np.append(L, [a[1] for a in y2_topN_theta])
				# L.extend([a[1] for a in y2_topN_theta])
				for a in sorted(y2_topN_theta, key=lambda a: a[0], reverse=True):
					y[a[1]] = np.full(y[0].shape, 0)
					y[a[1]][a[2]] = 1
					# y[a[1]] = a[2]		
					# U_.pop(a[0])
					if (a[0] < U_.shape[0]-1):
						U_ = np.append(U_[:a[0]], U_[a[0]+1:], axis=0)
					else:
						U_ = U_[:a[0]]
				num_to_add = len(y2_topN_theta)

			# print("len U: ", len(U_))
			# n, p = [], []
			
			# for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
			# 	#we added all that we needed to for this iteration, so break
			# 	if len(p) == 2 * self.p_ and len(n) == 2 * self.n_:
			# 		break

			# 	#update our newly 'labeled' samples.  Note that we are only 'labeling' a single sample
			# 	#with each inner iteration.  We want to add 2p + 2n samples per outer iteration, but classifiers must agree

			# 	if y1_i == y2_i == 1 and len(p) < self.p_:
			# 		p.append(i)

			# 	if y2_i == y1_i == 0 and len(n) < self.n_:
			# 		n.append(i)


			# #label the samples and remove thes newly added samples from U_
			# y[[U_[x] for x in p]] = 1
			# y[[U_[x] for x in n]] = 0

			# L.extend([U_[x] for x in p])
			# L.extend([U_[x] for x in n])

			#TODO: optimize these removals from U_
			#this is currently (2p + 2n)O(n)
			#and I think it can be reduced to O(n) rather easily
			# for i in p: U_.pop(i)
			# for i in n: U_.pop(i)

			#add new elements to U_
			add_counter = 0 #number we have added from U to U_
			# num_to_add = len(p) + len(n)
			while add_counter != num_to_add and U.shape[0]>0:
				add_counter += 1
				# print(U_.shape, U.shape)
				# assert U_.shape[1] == U.shape[1]
				U_ = np.append(U_, [U[0]], axis=0)
				U = U[1:]
				# U_.append(U.pop())


			# preds_val = self.predict_proba(validation_x1, validation_x2)
			# val_auc = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val, axis = 1))
			# self.epoch_val_auc[it] = val_auc
			val_error = self.test_error(validation_x1, validation_x2, validation_y)
			self.epoch_val_auc[it] = 1.0 - val_error

			# preds_train = self.predict_proba(X1, X2)
			# train_auc = accuracy_score(np.argmax(y, axis = 1), np.argmax(preds_train, axis = 1))
			# self.epoch_train_auc[it] = train_auc
			train_error = self.test_error(X1, X2, y)
			self.epoch_train_auc[it] = 1.0 - train_error


			#TODO: Handle the case where the classifiers fail to agree on any of the samples (i.e. both n and p are empty)


		#let's fit our final model
		self.clf1_.fit(X1[L], y[L], validation_x1, validation_y)
		self.clf2_.fit(X2[L], y[L], validation_x2, validation_y)

		print("epoch_val_auc:", self.epoch_val_auc)
		print("epoch_train_auc:", self.epoch_train_auc)


	#TODO: Move this outside of the class into a util file.
	def supports_proba(self, clf, x):
		"""Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
		try:
			clf.predict_proba([x])
			return True
		except:
			return False
	
	def predict(self, X1, X2):
		"""
		Predict the classes of the samples represented by the features in X1 and X2.
		Parameters:
		X1 - array-like (n_samples, n_features1)
		X2 - array-like (n_samples, n_features2)
		
		Output:
		y - array-like (n_samples)
			These are the predicted classes of each of the samples.  If the two classifiers, don't agree, we try
			to use predict_proba and take the classifier with the highest confidence and if predict_proba is not implemented, then we randomly
			assign either 0 or 1.  We hope to improve this in future releases.
		"""

		y1 = self.clf1_.predict(X1)
		y2 = self.clf2_.predict(X2)

		# proba_supported = self.supports_proba(self.clf1_, X1[0]) and self.supports_proba(self.clf2_, X2[0])

		#fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
		y_pred = np.asarray([-1] * X1.shape[0])

		print("cotrain predict start loop")

		y1_probs = self.clf1_.predict_proba(X1)
		y2_probs = self.clf2_.predict_proba(X2)
		y_pred_max = np.argmax(y1_probs + y2_probs, axis = 1)

		for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
			if y1_i == y2_i:
				# print("hellYEAH")
				y_pred[i] = y1_i
			# elif proba_supported:
			else:
				# y1_probs = self.clf1_.predict_proba([X1[i]])[0]
				# y2_probs = self.clf2_.predict_proba([X2[i]])[0]
				# print("shape: ", X1[i].shape)
				# print("X1[i]: ", X1[i])
				# y1_probs = self.clf1_.predict_proba(X1[i].reshape((1,27)))
				# y2_probs = self.clf2_.predict_proba(X2[i].reshape((1,27)))
				# sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
				# max_sum_prob = max(sum_y_probs)
				# y_pred[i] = sum_y_probs.index(max_sum_prob)
				y_pred[i] = y_pred_max[i]

			# else:
			# 	#the classifiers disagree and don't support probability, so we guess
			# 	print("guess")
			# 	y_pred[i] = random.randint(0, 8)

			
		#check that we did everything right
		assert not (-1 in y_pred)

		return y_pred


	def predict_proba(self, X1, X2):
		"""Predict the probability of the samples belonging to each class."""
		y_proba = np.full((X1.shape[0], 7), -1).astype(np.float64)

		y1_proba = self.clf1_.predict_proba(X1)
		y2_proba = self.clf2_.predict_proba(X2)

		# print("y1_proba", y1_proba)
		# a = y1_proba[0]
		# print(sum(a))
		# assert abs(sum(a) - 1) <= 0.0001

		for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
			y_proba[i] = (y1_i_dist + y2_i_dist) / 2
			# y_proba[i] = 
			# print(y1_i_dist, y2_i_dist, y_proba[i])
			# print(np.argmax(y1_i_dist), np.argmax(y2_i_dist), np.argmax(y_proba[i]))
			# y_proba[i] = np.maximum(y1_i_dist, y2_i_dist)
			# y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
			# y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

		# print("clf1 clf2 equal %", np.sum(np.equal(np.argmax(y1_proba, axis=1), np.argmax(y2_proba, axis=1))) / X1.shape[0])
		# print("clf1 y_proba equal %", np.sum(np.equal(np.argmax(y1_proba, axis=1), np.argmax(y_proba, axis=1))) / X1.shape[0])
		# print("clf2 y_proba equal %", np.sum(np.equal(np.argmax(y2_proba, axis=1), np.argmax(y_proba, axis=1))) / X1.shape[0])

		_epsilon = 0.0001
		# assert all(abs(sum(y_dist) - 1) <= _epsilon for y_dist in y_proba)
		return y_proba


	def test_error(self, X1, X2, Y):
		y_hat = self.predict(X1, X2)
		diff = np.equal(y_hat, np.argmax(Y, axis=1).astype(int))
		diff_sum = np.sum(diff)
		avg_error = 1 - diff_sum / np.shape(X1)[0]
		return avg_error