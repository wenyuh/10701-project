
from Quasi_Newton_S3VM import Quasi_Newton_S3VM as QN_S3VM
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def load_data():
    data_train = np.genfromtxt('forest_sub.csv', dtype=int, delimiter=',')

    #print(data_train.shape)
    index_all = np.arange(len(data_train))
    np.random.shuffle(index_all)
    print(index_all)
    data_train = data_train[index_all, :]
    data_train = data_train[1:, :]
    data_train = data_train[:, 1:]
    
    train_x = data_train[:11340, :-1].astype(np.float64)
    train_x = preprocessing.scale(train_x)
    train_y = data_train[:11340, -1].astype(np.float64)
    validation_x = data_train[(11340):(11340+3780), :-1].astype(np.float64)
    validation_x = preprocessing.scale(validation_x)
    validation_y = data_train[(11340):(11340+3780), -1].astype(np.float64)
    test_x = data_train[(11340+3780):31340, :-1].astype(np.float64)
    test_x = preprocessing.scale(test_x)
    test_y = data_train[(11340+3780):31340, -1].astype(np.float64)



    return {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
        'validation_x': validation_x,
        'validation_y': validation_y
    }

data = load_data()
train_x = data['train_x']
train_y = data['train_y']
validation_x = data['validation_x']
validation_y = data['validation_y']
test_x = data['test_x']
test_y = data['test_y']

labels = [1, 2, 3, 4, 5, 6, 7]

index = np.arange(len(train_y))
#np.random.seed(1)
#np.random.shuffle(index)


label_idx = index[:int(np.ceil(len(train_y) * 0.2))]
unlabel_idx = index[int(np.ceil(len(train_y) * 0.2)):]

labeled_x = train_x[label_idx, :]
labeled_y = train_y[label_idx]
unlabeld_x = train_x[unlabel_idx, :]
unlabeld_y = train_y[unlabel_idx]

print(labeled_y)
print(validation_y)
print(unlabeld_y)
print(test_y)



#for sigma in [0.5]:
prediction_mat = np.zeros((len(labels)-1, len(unlabeld_x)))
y_mat = np.zeros((len(labels)-1, len(unlabeld_x)))

prediction_val = np.zeros((len(labels)-1, len(validation_x)))
y_mat_val = np.zeros((len(labels)-1, len(validation_x)))

prediction_test = np.zeros((len(labels)-1, len(test_x)))
y_mat_test = np.zeros((len(labels)-1, len(test_x)))


for i in range(len(labels) - 1):
#for i in range(1):
    curr_label = labels[i]
    idx = np.where(labeled_y == curr_label)
    curr_y = np.zeros(len(labeled_y))
    curr_y -= 1
    curr_y[idx] = 1
    tsvm = QN_S3VM(labeled_x, curr_y, unlabeld_x)
    tsvm.fit()
    pred, y = tsvm.get_predictions(unlabeld_x)
    prediction_mat[i] = pred
    y_mat[i] = y
    y[y == 1] = curr_label
    y[y != 1] = 0
    print(accuracy_score(y, unlabeld_y))

    pred_val, y_val = tsvm.get_predictions(validation_x)
    prediction_val[i] = pred_val
    y_mat_val[i] = y_val
    y_val[y_val == 1] = curr_label
    y_val[y_val != 1] = 0
    print(accuracy_score(y_val, validation_y))

    pred_test, y_test = tsvm.get_predictions(test_x)
    prediction_test[i] = pred_test
    y_mat_test[i] = y_test
    y_test[y_test == 1] = curr_label
    y_test[y_test != 1] = 0
    print(accuracy_score(y_test, test_y))


y_mat_r = np.sum(y_mat, axis = 0)
res = np.zeros(len(y_mat_r))
for i in range(len(y_mat_r)):
    if y_mat_r[i] == 0:
        res[i] = 7
    else:
        res[i] = np.argmax(y_mat[:, i]) + 1
print(accuracy_score(res, unlabeld_y))

y_val_r = np.sum(y_mat_val, axis = 0)
res_val = np.zeros(len(y_val_r))
for i in range(len(y_val_r)):
    if y_val_r[i] == 0:
        res_val[i] = 7
    else:
        res_val[i] = np.argmax(y_mat_val[:, i]) + 1
print(accuracy_score(res_val, validation_y))


y_test_r = np.sum(y_mat_test, axis = 0)
res_test = np.zeros(len(y_test_r))
for i in range(len(y_test_r)):
    if y_test_r[i] == 0:
        res_test[i] = 7
    else:
        res_test[i] = np.argmax(y_mat_test[:, i]) + 1
print(accuracy_score(res_test, test_y))



