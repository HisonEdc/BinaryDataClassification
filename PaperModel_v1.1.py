"""
author: He Jiaxin
Date: 29/03/2019
Version: V1.1
Function: Realize Paper Model in MNIST, classifying 0 & 5
"""

# import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import random

# set global variable
m = 100
p = 50

# get MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('..../MNIST_data',one_hot=True)

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
sub_train_class0 = train_class0[:p, :].T   # (784, 100)
sub_train_class1 = train_class1[:p, :].T   # (784, 100)

# generate kernel matrix A (m by n), which n = 784, and here we choose m = 50
# take A to have independent identically distributed standard Gaussian entries
A = np.random.randn(m, 784)

# inner product of A and X, then use sigmoid function to identify all elements as 0 or 1
Q_class0 = np.dot(A, sub_train_class0)  # (50, 100)
Q_class1 = np.dot(A, sub_train_class1)  # (50, 100)

Q_class0 = np.where(Q_class0 >= 0, 1, -1)
Q_class1 = np.where(Q_class1 >= 0, 1, -1)

Q = [Q_class0, Q_class1]

# construct L layers, each layer has m sets
# here, as an example, we choice L = 2, and according to the historical code, m = 50
# for the first layer, it is easy to operate, and we randomly arrange m = 50 rows
layer1 = list(range(m))
np.random.shuffle(layer1)

# first, we list all possible combination of 2C50, then randomly choose 50
all_combination_layer2 = []
for i in itertools.combinations(range(m), 2):
    all_combination_layer2.append(i)   # note the type of elements is tuple
layer2 = random.sample(all_combination_layer2, m)

# the next step we need to do is to calculate r(l, i, t, g)
# this index has 4 dimensions, we need to calculate r for different layers, selection sets, sign patterns and class
# as different layers have different numbers of sign pattern, so we divide r into the layers'number(2) sets
r_layer1 = np.zeros((m, 2, 2))  # the parameter is 'i' sets, 't' sign patterns, and 'g' classes
r_layer2 = np.zeros((m, 4, 2))

# p is the number of training points from 'i'-th set, 't'-th sign pattern, and 'g'-th class
p_layer1 = np.zeros((m, 2, 2))
p_layer2 = np.zeros((m, 4, 2))

# fill r_layer1 matrix
def count_numbers_layer1(set, pattern, clas):
    lambda_layer1_set = Q[clas][layer1[set]]
    if pattern == 0:
        return list(lambda_layer1_set).count(-1) + 1
    else:
        return list(lambda_layer1_set).count(1) + 1

for i in range(m):
    for t in range(2):
        for g in range(2):
            p_layer1[i, t, g] = count_numbers_layer1(i, t, g)

for i in range(m):
    for t in range(2):
        for g in range(2):
            minus = []
            for j in range(2):
                minus.append(abs(p_layer1[i,t,g] - p_layer1[i,t,j]))
            r_layer1[i, t, g] = p_layer1[i, t, g] * sum(minus) / np.power(sum(p_layer1[i, t]), 2)

# fill r_layer2 matrix
def count_numbers_layer2(set, pattern, clas):
    lambda_layer2_set = Q[clas][list(layer2[set])].T
    count = 1
    if pattern == 0:
        for i in range(len(lambda_layer2_set)):
            if (lambda_layer2_set[i] == [-1, -1]).all():
                count += 1
    elif pattern == 1:
        for i in range(len(lambda_layer2_set)):
            if (lambda_layer2_set[i] == [-1, 1]).all():
                count += 1

    elif pattern == 2:
        for i in range(len(lambda_layer2_set)):
            if (lambda_layer2_set[i] == [1, -1]).all():
                count += 1
    else:
        for i in range(len(lambda_layer2_set)):
            if (lambda_layer2_set[i] == [1, 1]).all():
                count += 1
    return count

for i in range(m):
    for t in range(4):
        for g in range(2):
            p_layer2[i, t, g] = count_numbers_layer2(i, t, g)

for i in range(m):
    for t in range(4):
        for g in range(2):
            minus = []
            for j in range(2):
                minus.append(abs(p_layer2[i, t, g] - p_layer2[i, t, j]))
            r_layer2[i, t, g] = p_layer2[i, t, g] * sum(minus) / np.power(sum(p_layer2[i, t]), 2)



# classification: testing
# for this example, we classify 50 images per digit class

test_class0 = test_data[np.where(test_labels[:, 0] == 1)]   # (980, 784)
test_class1 = test_data[np.where(test_labels[:, 5] == 1)]   # (1135, 784)
test_class2 = test_data[np.where(test_labels[:, 2] == 1)]   # (1032, 784)

# considering only 2 digit classes: 0 and 1, each for 100 samples
# randomly get model dataset class0 and class1 from train_data
sub_test_class0 = test_class0[:50, :].T   # (784, 50)
sub_test_class1 = test_class1[:50, :].T   # (784, 50)

# inner product of A and X, then use sigmoid function to identify all elements as 0 or 1
Q_test_class0 = np.dot(A, sub_test_class0)  # (50, 50)
Q_test_class1 = np.dot(A, sub_test_class1)  # (50, 50)

Q_test_class0 = np.where(Q_test_class0 >= 0, 1, -1)
Q_test_class1 = np.where(Q_test_class1 >= 0, 1, -1)

Q_test = np.hstack((Q_test_class0, Q_test_class1))  # (50,100)

test_predict = np.zeros((100, 2))
for k in range(Q_test.shape[1]):
    sample = Q_test[:, k]
    for i in range(len(Q_test)):
        # accumulate b value of layer1
        if sample[layer1[i]] == -1:
            test_predict[k] += r_layer1[i, 0]
        else:
            test_predict[k] += r_layer1[i, 1]
        # accumulate b value of layer2
        if (sample[list(layer2[i])] == [-1, -1]).all():
            test_predict[k] += r_layer2[i, 0]
        elif (sample[list(layer2[i])] == [-1, 1]).all():
            test_predict[k] += r_layer2[i, 1]
        elif (sample[list(layer2[i])] == [1, -1]).all():
            test_predict[k] += r_layer2[i, 2]
        else:
            test_predict[k] += r_layer2[i, 3]

test_predict_class = np.argmax(test_predict, axis=1)

true_class = [0] * 50 + [1] * 50
mistake_count = 0
for i in range(100):
    if test_predict_class[i] != true_class[i]:
        mistake_count += 1
print('the number of error label is {}'.format(mistake_count))
