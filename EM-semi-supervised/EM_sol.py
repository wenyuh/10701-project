from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


def load_data():
    data_train = np.genfromtxt('covtype.data', dtype=int, delimiter=',')
    
    train_x = data_train[:11340, :-1].astype(np.float64)
    train_y = data_train[:11340, -1].astype(np.float64)
    validation_x = data_train[(11340):(11340+3780), :-1].astype(np.float64)
    validation_y = data_train[(11340):(11340+3780), -1].astype(np.float64)
    test_x = data_train[(11340+3780):, :-1].astype(np.float64)
    test_y = data_train[(11340+3780):, -1].astype(np.float64)


    # train_x = data_train[:100, :-1].astype(np.float64)
    # train_y = data_train[:100, -1].astype(np.float64)
    # validation_x = data_train[(100):(150), :-1].astype(np.float64)
    # validation_y = data_train[(100):(150), -1].astype(np.float64)
    # test_x = data_train[(200):250, :-1].astype(np.float64)
    # test_y = data_train[200:250, -1].astype(np.float64)


    return {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
        'validation_x': validation_x,
        'validation_y': validation_y
    }

# data_set = load_data()
# x_supervised = train_x[:11000, :]
# y_supervised = train_y[:11000, :]
# x_unsupervised = train_x[11000:11340, :]


class NaiveBayesSemiSupervised(object):
	def __init__(self, data_set, max_rounds=30, tolerance=1e-6):
		self.data_set = data_set
		# self.max_features = np.shape(train_x)[1]
		self.max_features = 54
		self.n_labels = 7
		self.max_rounds = max_rounds
		self.tolerance = tolerance
		print("init")
        

	def train(self, x_supervised, x_unsupervised, y_supervised):
        
		"""
		train the modified Naive bayes classifier using both labelled and 
		unlabelled data. We use the CountVectorizer vectorizaton method from scikit-learn

		positional arguments:!
		    
		    -- X_supervised: list of documents (string objects). these documents have labels
		    
		        example: ["all parrots are interesting", "some parrots are green", "some parrots can talk"]
		        
		    -- X_unsupervised: list of documents (string objects) as X_supervised, but without labels
		    
		    -- y_supervised: labels of the X_supervised documents. list or numpy array of integers. 
		    
		        example: [2, 0, 1, 0, 1, ..., 0, 2]
		"""

		# prior = np.array([1.0/7.0]*7)
		# print(prior)
		# prior = np.array
		# print("prior", prior)
		# clf = GaussianNB()
		clf = BernoulliNB()
		clf.fit(x_supervised, y_supervised)
		# self.train_naive_bayes(X_supervised, y_supervised)

		predi = clf.predict(x_supervised)

		old_likelihood = 1

		while self.max_rounds > 0:
		    
			self.max_rounds -= 1
			# E-step
			predi = clf.predict(x_unsupervised)
			# M-step
			clf.fit(x_unsupervised, predi)
			# calculate new total likelihood
			predi = clf.predict(x_supervised)
			unsupervised_log_matrix = clf._joint_log_likelihood(x_unsupervised)
			supervised_log_matrix = clf._joint_log_likelihood(x_supervised)
			print("unsupervised_log_matrix before log", unsupervised_log_matrix)
			# unsupervised_log_matrix = np.log(unsupervised_log_matrix)
			# supervised_log_matrix = np.log(supervised_log_matrix)
			# print("unsupervised_log_matrix", unsupervised_log_matrix)
			total_likelihood = self.get_log_likelihood(unsupervised_log_matrix, supervised_log_matrix, y_supervised)
			print("total likelihood: {}".format(total_likelihood))

			if self._stopping_time(old_likelihood, total_likelihood):
			    
				break

			old_likelihood = total_likelihood.copy()
		self.clf = clf

	def predict(self, x_test):
		return self.clf.predict(x_test)

	def test_error(self, x_test, y_test):
		print(x_test)
		y_hat = self.clf.predict(x_test)
		print(y_hat)
		print(y_test)
		diff = np.equal(y_hat, y_test).astype(int)
		diff_sum = np.sum(diff)
		avg_error = 1 - diff_sum / np.shape(x_test)[0]
		return avg_error

		# y_hat = (self.clf).predict(x_test)
		# print(type(y_hat))
		# print(y_hat)
		# print(y_test)
		# diff = np.equal[y_hat, y_test].astype(int)
		# print(diff)
		# res = diff / (np.shape(x_test)[0])
		# print("test error:", res)
		# return res




	def get_log_likelihood(self, unsupervised_log_matrix, supervised_log_matrix, y_supervised):
	        
		"""
		returns the total log-likelihood of the model, taking into account unsupervised data

		positional arguments:
			-- unsupervised_log_matrix: log likelihood of unsupervised x [N_unsup, C]

			-- supervised_log_matrix: log likelihood of supervised x [N_sup, C]
		    
		    -- X_supervised: list of documents (string objects). these documents have labels
		    
		        example: ["all parrots are interesting", "some parrots are green", "some parrots can talk"]
		        
		    -- X_unsupervised: list of documents (string objects) as X_supervised, but without labels
		    
		    -- y_supervised: labels of the X_supervised documents. list or numpy array of integers. 
		    
		        example: [2, 0, 1, 0, 1, ..., 0, 2]

		"""
		assert np.shape(unsupervised_log_matrix)[1] == 7
		assert np.shape(supervised_log_matrix)[1] == 7

		print(np.shape(unsupervised_log_matrix))
		print(np.shape(supervised_log_matrix))
		print(unsupervised_log_matrix)
		unsupervised_term = np.sum(np.amax(unsupervised_log_matrix, axis=1))
		N_sup = np.shape(supervised_log_matrix)[0] 
		y_supervised = y_supervised.astype(int)


		supervised_term = np.sum(supervised_log_matrix[np.arange(N_sup), y_supervised-1])
		total_likelihood = supervised_term + unsupervised_term

		return total_likelihood


	def _stopping_time(self, old_likelihood, new_likelihood):
        
		"""
		returns True if there is no significant improvement in log-likelihood and false else

		positional arguments:
		    
		    -- old_likelihood: log-likelihood for previous iteration
		    
		    -- new_likelihood: new log-likelihood

		"""
		relative_change = np.absolute((new_likelihood-old_likelihood)/new_likelihood) 

		if (relative_change < self.tolerance):

			print("stopping time")
			return True

		else:
		    
			return False


# data_set = load_data()
# x_supervised = np.loadtxt("train_x.txt")
# y_supervised = np.loadtxt("train_y.txt")
# x_unsupervised = np.loadtxt("validation_x.txt")
# x_supervised = train_x[:11340, :]
# y_supervised = train_y[:11340, :]
# x_unsupervised = train_x[11340:(11340+3780), :]


# data_set = load_data()
# x_supervised = train_x[:50, :]
# y_supervised = train_y[:50, :]
# x_unsupervised = train_x[50:, :]
# print(x_supervised, flush=True)

