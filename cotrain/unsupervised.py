import scipy.io
import numpy as np

from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from scipy.stats import multivariate_normal


def load_data():
    data_train = np.genfromtxt('forest_sub.csv', dtype=int, delimiter=',')

    print(data_train.shape)
    index = np.arange(len(data_train))
    np.random.shuffle(index)
    
    train_x = data_train[:11340, :-1].astype(np.float64)
    train_y = data_train[:11340, -1].astype(np.float64)
    train_x = train_x[1:, :]
    train_x = train_x[:, 1:]
    train_y = train_y[1:]
    #validation_x = data_train[(11340):(11340+3780), :-1].astype(np.float64)
    #validation_y = data_train[(11340):(11340+3780), -1].astype(np.float64)
    #test_x = data_train[(11340+3780):31340, :-1].astype(np.float64)
    #test_y = data_train[(11340+3780):31340, -1].astype(np.float64)



    return {
        'train_x': train_x,
        'train_y': train_y
#        'test_x': test_x,
#        'test_y': test_y,
#        'validation_x': validation_x,
#        'validation_y': validation_y
    }

data = load_data()
train_x = data['train_x']
train_y = data['train_y']

X = train_x[:3000, :]
y = train_y[:3000].astype(int)


for i in range(len(X[0])):
	for j in range(i+1, len(X[0])):
		feature_1 = X[:, i]
		feature_2 = X[:, j]
		fig = plt.figure()
		plt.scatter(feature_1, feature_2, c=y, marker = '.')
		plt.xlabel(i)
		plt.ylabel(j)
		plt.show()


#pca = PCA()
#data_scaled = preprocessing.scale(X)
#pca.fit(data_scaled)
#score = pca.transform(data_scaled)

#variance = pca.explained_variance_
#ratio = pca.explained_variance_ratio_
#ratio_sum = pca.explained_variance_ratio_.cumsum()
#variance_sum = pca.explained_variance_.cumsum()

#fig = plt.figure()
#plt.plot(ratio_sum)
#fig.suptitle('total variance explanined by the first k components', fontsize=14)
#plt.xlabel('number of principal components', fontsize=14)
#plt.ylabel('variance explained', fontsize=14)
#plt.savefig("pca_variance_winsize10") 
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#y = train_y[:3000].astype(int)
#ax.scatter(score[:, 0], score[:, 1], score[:, 2], c=y, marker='.')

#sns_plot = sns.scatterplot(score[:, 0], score[:, 1], hue = train_y.astype(int))

#plt.show()
