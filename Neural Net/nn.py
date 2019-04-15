import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape
from keras.layers.merge import concatenate, dot
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD

import matplotlib.pyplot as plt

def load_data(data):
    data_train = np.genfromtxt(data, dtype=int, delimiter=',')

    #print(data_train.shape)
    index_all = np.arange(len(data_train))
    np.random.shuffle(index_all)

    data_train = data_train[index_all, :]
    data_train = data_train[1:, :]
    
    train_x = data_train[:11340, :-1].astype(np.float64)
    train_x = preprocessing.scale(train_x)
    train_y = data_train[:11340, -1].astype(str)
    validation_x = data_train[(11340):(11340+3780), :-1].astype(np.float64)
    validation_x = preprocessing.scale(validation_x)
    validation_y = data_train[(11340):(11340+3780), -1].astype(str)
    test_x = data_train[(11340+3780):31340, :-1].astype(np.float64)
    test_x = preprocessing.scale(test_x)
    test_y = data_train[(11340+3780):31340, -1].astype(str)



    return {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
        'validation_x': validation_x,
        'validation_y': validation_y
    }


def get_model():
    '''
    define model
    '''

    model = Sequential()

    model.add(Dense(100, input_shape=(54, ), init='RandomUniform', activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    opt = RMSprop(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    return model

def train():
    model = get_model()

    data = load_data('covtype.data')
    train_x = data['train_x']
    train_y = data['train_y']
    validation_x = data['validation_x']
    validation_y = data['validation_y']
    test_x = data['test_x']
    test_y = data['test_y']

    
    
    onehot_encoder = OneHotEncoder(sparse=False)
    train_y = onehot_encoder.fit_transform(train_y.reshape(len(train_y), 1))
    validation_y = onehot_encoder.fit_transform(validation_y.reshape(len(validation_y), 1))
    test_y = onehot_encoder.fit_transform(test_y.reshape(len(test_y), 1))

    index = np.arange(len(train_x))
    label_idx = index[:int(np.ceil(len(train_y) * 0.2))]

    train_x = train_x[label_idx, :]
    train_y = train_y[label_idx]

    early_stopping = EarlyStopping(monitor='val_acc', patience=5)

    hist = model.fit(train_x, train_y, validation_data=(validation_x, validation_y), 
    epochs=100, shuffle=True, callbacks=[early_stopping])


    preds_val = model.predict(validation_x)
    val_auc = accuracy_score(np.argmax(validation_y, axis = 1), np.argmax(preds_val, axis = 1))

    print("val_auc", val_auc)

    preds_test = model.predict(test_x, verbose=1)
    test_auc = accuracy_score(np.argmax(test_y, axis = 1), np.argmax(preds_test, axis = 1))
    
    print("test_auc", test_auc)


train()
