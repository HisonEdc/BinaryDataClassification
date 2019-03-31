"""
author: He Jiaxin
Date: 31/03/2019
Version: V5.0
Function: compare with other traditional model
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

# set global variable
m = 100
p = 50

# get MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/Users/hejiaxin/PycharmProjects/MNIST_data',one_hot=True)

# count the sample number of each datasets
train_nums = mnist.train.num_examples
validation_nums = mnist.validation.num_examples
test_nums = mnist.test.num_examples

# get train_data,validation_data,test_data
train_data = mnist.train.images     # (55000, 784)
val_data = mnist.validation.images  # (5000, 784)
test_data = mnist.test.images       # (10000, 784)

# get each dataset's label, which is [0,1,...,0,0], 1 by 10 matrix
train_labels = mnist.train.labels     #(55000,10)
val_labels = mnist.validation.labels  #(5000,10)
test_labels = mnist.test.labels       #(10000,10)

# for this code, we simply realize the model of this paper
# get model dataset class0-class9 from train_data
train_class0 = train_data[np.where(train_labels[:, 0] == 1)]   # (5444, 784)
train_class1 = train_data[np.where(train_labels[:, 5] == 1)]   # (6179, 784)

# considering only 2 digit classes: 0 and 1, each for 100 samples
# randomly get model dataset class0 and class1 from train_data
sub_train_class0 = train_class0[:p, :]   # (100, 784)
sub_train_class1 = train_class1[:p, :]   # (100, 784)
train_set = np.vstack((sub_train_class0, sub_train_class1))

# classification: testing
# for this example, we classify 50 images per digit class
test_class0 = test_data[np.where(test_labels[:, 0] == 1)]  # (980, 784)
test_class1 = test_data[np.where(test_labels[:, 5] == 1)]  # (1135, 784)

# considering only 2 digit classes: 0 and 1, each for 100 samples
# randomly get model dataset class0 and class1 from train_data
sub_test_class0 = test_class0[:50, :]  # (50, 784)
sub_test_class1 = test_class1[:50, :]  # (50, 784)
test_set = np.vstack((sub_test_class0, sub_test_class1))

# create training and testing labels
train_class = [0] * 50 + [1] * 50
test_class = [0] * 50 + [1] * 50

# SVC
initial_time = time.time()
clf_svc = svm.SVC(gamma='scale')
clf_svc.fit(train_set, train_class)
training_time = time.time()-initial_time
print("Training Time = ", training_time)
accuracy = clf_svc.score(test_set, test_class)
print("SVM accuracy =", accuracy)

# random forest
initial_time = time.time()
clf_rf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf_rf, train_set, train_class, cv=5)
training_time = time.time()-initial_time
print("Training Time = ", training_time)
print("Random Forest accuracy =", scores.mean())

# CNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras import optimizers
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# global variable
batch_size = 60
nb_classes = 2  # number of classes
epochs = 50  # iterative times
img_rows, img_cols = 28, 28  # image size
nb_filters = 32  # number of kernel
pool_size = (2, 2)  # pooling size
kernel_size = (5, 5)  # kernel size
input_shape = (img_rows, img_cols, 1)

x_train = train_set.reshape(2 * p, 28, 28)
x_train = x_train[:, :, :, np.newaxis]
y_train = np.array([[1, 0]] * p + [[0, 1]] * p)
x_test = test_set.reshape(100, 28, 28)
x_test = x_test[:, :, :, np.newaxis]
y_test = np.array([[1, 0]] * 50 + [[0, 1]] *50)

val_class0 = val_data[np.where(val_labels[:, 0] == 1)]
val_class1 = val_data[np.where(val_labels[:, 5] == 1)]
sub_val_class0 = val_class0[:int(p / 10), :]
sub_val_class1 = val_class1[:int(p / 10), :]
val_set = np.vstack((sub_val_class0, sub_val_class1))
x_val = val_set.reshape(int(2 * p / 10), 28, 28)
x_val = x_val[:, :, :, np.newaxis]
y_val = np.array([[1, 0]] * int(p / 10) + [[0, 1]] * int(p / 10))


# build CNN model
model = Sequential()
model.add(Conv2D(6, kernel_size, input_shape=input_shape,strides=1))  # convolution layer1
model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # pooling layer1
model.add(Conv2D(12, kernel_size, strides=1))  # convolution layer2
model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # pooling layer2
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

sgd = optimizers.SGD(lr=0.01, momentum=0.5)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
