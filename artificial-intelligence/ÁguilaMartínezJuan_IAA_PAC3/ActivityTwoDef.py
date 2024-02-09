# -*- coding: utf-8 -*-

"""
WHOLESALES CUSTOMERS
Learning algorithm: kNN / Naïve Bayes / Decision Tree / SVM (multiple kernels)
With variable normalization
No Dimensionality reductionsingle-validation
Training set size = 4 x test size
No statistical test
"""

from random import shuffle
import numpy
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm 

from sklearn import metrics
from sklearn.externals.six import StringIO  

from sklearn import preprocessing

import matplotlib.pyplot as plt

# open file and load data into a list of lists
l = list(map(lambda l: (l.strip()).split(','), 
             open('Data/Wholesale Customers.csv', 'r').readlines()))
                 
# delete head
del(l[0])
                 
# examples shuffle                  
shuffle(l)                

# Split thelist in Labels + Data
data = []
target = []
for x in l:
    data.append(x[2:])
    target.append(x[:2])
    # target.append(x[0])

X = numpy.array(data)
y = numpy.array(target)

X = X.astype(numpy.float)
X = preprocessing.normalize(X)

y = y.astype(numpy.float)

train_n = 0.6
train_size = math.floor(len(l) * train_n)

train_features = X[:train_size]
train_labels = y[:train_size]

test_features = X[train_size:]
test_labels = y[train_size:]
   
# Train a kNN Classifier
   
k = 7

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_features, train_labels)
y_pred = knn.predict(test_features)

n_test_labels = 0.0
first_order_acc = 0.0
second_order_acc = 0.0

for pred, real in zip(y_pred, test_labels):
    n_test_labels = n_test_labels + 1
    if pred[0] == real[0]:
        first_order_acc = first_order_acc + 1
        if pred[1] == real[1]:
            second_order_acc = second_order_acc + 1
            
first_order_error = 100 * (n_test_labels - first_order_acc) / n_test_labels
total_error = 100 * (n_test_labels - second_order_acc) / n_test_labels

# single-validation
print("kNN Single-validation")
print("%d Neighbours. Size = %d pct of dataset." % (k, train_n * 100))
print("First step Error. %d pct. Second step Error: %d pct" % (first_order_error, total_error))

# Train a Decision Tree Classifier
   
dt = tree.DecisionTreeClassifier()   
dt.fit(train_features, train_labels)
y_pred = dt.predict(test_features)

n_test_labels = 0.0
first_order_acc = 0.0
second_order_acc = 0.0

for pred, real in zip(y_pred, test_labels):
    n_test_labels = n_test_labels + 1
    if pred[0] == real[0]:
        first_order_acc = first_order_acc + 1
        if pred[1] == real[1]:
            second_order_acc = second_order_acc + 1
            
first_order_error = 100 * (n_test_labels - first_order_acc) / n_test_labels
total_error = 100 * (n_test_labels - second_order_acc) / n_test_labels

# single-validation
print("Decision Tree Single-validation")
print("Size = %d pct of dataset." % (train_n * 100))
print("First step Error. %d pct. Second step Error: %d pct" % (first_order_error, total_error))

# Train a Gaussian Naive Bayes Classifier
   
nb = GaussianNB()
nb.fit(train_features, train_labels[:,0])
y1_pred = nb.predict(test_features)

nb.fit(train_features, train_labels[:,1])
y2_pred = nb.predict(test_features)

n_test_labels = 0.0
first_order_acc = 0.0
second_order_acc = 0.0

for pred1, pred2, real in zip(y1_pred, y2_pred, test_labels):
    n_test_labels = n_test_labels + 1
    if pred1 == real[0]:
        first_order_acc = first_order_acc + 1
        if pred2 == real[1]:
            second_order_acc = second_order_acc + 1
            
first_order_error = 100 * (n_test_labels - first_order_acc) / n_test_labels
total_error = 100 * (n_test_labels - second_order_acc) / n_test_labels

# single-validation
print("Decision Tree Single-validation")
print("Size = %d pct of dataset." % (train_n * 100))
print("First step Error. %d pct. Second step Error: %d pct" % (first_order_error, total_error))
    
# Train a SVM Clasifier
    
linear_svc = svm.SVC(kernel="linear")

linear_svc.fit(train_features, train_labels[:,0])
y1_pred = linear_svc.predict(test_features)

linear_svc.fit(train_features, train_labels[:,1])
y2_pred = linear_svc.predict(test_features)

n_test_labels = 0.0
first_order_acc = 0.0
second_order_acc = 0.0

for pred1, pred2, real in zip(y1_pred, y2_pred, test_labels):
    n_test_labels = n_test_labels + 1
    if pred1 == real[0]:
        first_order_acc = first_order_acc + 1
        if pred2 == real[1]:
            second_order_acc = second_order_acc + 1
            
first_order_error = 100 * (n_test_labels - first_order_acc) / n_test_labels
total_error = 100 * (n_test_labels - second_order_acc) / n_test_labels

# single-validation
print("Linear SVM Single-validation")
print("Size = %d pct of dataset." % (train_n * 100))
print("First step Error. %d pct. Second step Error: %d pct" % (first_order_error, total_error))

rbf_svc = svm.SVC(kernel="rbf")

rbf_svc.fit(train_features, train_labels[:,0])
y1_pred = rbf_svc.predict(test_features)

rbf_svc.fit(train_features, train_labels[:,1])
y2_pred = rbf_svc.predict(test_features)

n_test_labels = 0.0
first_order_acc = 0.0
second_order_acc = 0.0

for pred1, pred2, real in zip(y1_pred, y2_pred, test_labels):
    n_test_labels = n_test_labels + 1
    if pred1 == real[0]:
        first_order_acc = first_order_acc + 1
        if pred2 == real[1]:
            second_order_acc = second_order_acc + 1
            
first_order_error = 100 * (n_test_labels - first_order_acc) / n_test_labels
total_error = 100 * (n_test_labels - second_order_acc) / n_test_labels

# single-validation
print("Radial SVM Single-validation")
print("Size = %d pct of dataset." % (train_n * 100))
print("First step Error. %d pct. Second step Error: %d pct" % (first_order_error, total_error))

poly_svc = svm.SVC(kernel="poly")

poly_svc.fit(train_features, train_labels[:,0])
y1_pred = poly_svc.predict(test_features)

poly_svc.fit(train_features, train_labels[:,1])
y2_pred = poly_svc.predict(test_features)

n_test_labels = 0.0
first_order_acc = 0.0
second_order_acc = 0.0

for pred1, pred2, real in zip(y1_pred, y2_pred, test_labels):
    n_test_labels = n_test_labels + 1
    if pred1 == real[0]:
        first_order_acc = first_order_acc + 1
        if pred2 == real[1]:
            second_order_acc = second_order_acc + 1
            
first_order_error = 100 * (n_test_labels - first_order_acc) / n_test_labels
total_error = 100 * (n_test_labels - second_order_acc) / n_test_labels

# single-validation
print("Polynomic SVM Single-validation")
print("Size = %d pct of dataset." % (train_n * 100))
print("First step Error. %d pct. Second step Error: %d pct" % (first_order_error, total_error))

    

   