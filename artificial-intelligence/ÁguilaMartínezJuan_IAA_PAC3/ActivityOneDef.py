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
    target.append(x[0] + x[1])
    # target.append(x[0])

X = numpy.array(data)
y = numpy.array(target)

X = X.astype(numpy.float)
X = preprocessing.normalize(X)

y = y.astype(numpy.float)
    
array = range(10, 90) # pct of the training set over the total dataset

pack =[]
neighbours = [1, 3, 5, 7, 9, 11]

labels = ("1 neighbor", "3 neighbors", 
              "5 neighbors", "7 neighbors", "9 neighbors", "11 neighbors")

for i in array:
    accuracy = []

    # Train a k Nearest Neighbours Classifier
    for k in neighbours:    
        
        # train_n = 1/float(i)
        train_n = float(i) / 100
    
        train_size = math.floor(len(l) * train_n)
        
        train_features = X[:train_size]
        train_labels = y[:train_size]
        
        test_features = X[train_size:]
        test_labels = y[train_size:]    
        
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = knn.fit(train_features, train_labels).predict(test_features)
        acc = 100 * knn.score(test_features, test_labels)
    
        # Query the accuracy of the model with the test set
        accuracy.append(acc)
        
    pack.append(accuracy)  
    
P = numpy.array(pack)

plt.xlabel('% of training set')
plt.ylabel('Accuracy')

for i, l in zip(range(6), labels):
    print l
    plt.plot(array, P[:, i], label=l)
    
plt.legend(loc='upper left')
plt.show()

# Train a kNN Classifier

train_n = 0.6
train_size = math.floor(len(l) * train_n)

train_features = X[:train_size]
train_labels = y[:train_size]

test_features = X[train_size:]
test_labels = y[train_size:]

k = 7

knn = KNeighborsClassifier(n_neighbors=7)
y_pred = knn.fit(train_features, train_labels).predict(test_features)
acc = 100 * knn.score(test_features, test_labels)
# single-validation
print("kNN Single-validation")
print("%d Neighbours. Size = %d pct of dataset. Number of mislabeled points out of a total %d points : %d"
            % (k, train_n * 100, test_features.shape[0],(test_labels != y_pred).sum()))
print("Accuracy: %d pct"
            % (acc))

# Eval a decision tree
array = range(10, 90) # pct of the training set over the total dataset
pack =[]
for i in array:
    accuracy = []
        
    train_n = float(i) / 100
    
    train_size = math.floor(len(l) * train_n)
        
    train_features = X[:train_size]
    train_labels = y[:train_size]
        
    test_features = X[train_size:]
    test_labels = y[train_size:]    
        
    dt = tree.DecisionTreeClassifier()   
    y_pred = dt.fit(train_features, train_labels).predict(test_features)
    acc = 100 * dt.score(test_features, test_labels)
    
    # Query the accuracy of the model with the test set
    accuracy.append(acc)
        
    pack.append(accuracy)  
    
P = numpy.array(pack)

plt.xlabel('% of training set')
plt.ylabel('Accuracy')

for i, l in zip(range(6), labels):
    print l
    plt.plot(array, P[:, i], label=l)
    
plt.legend(loc='upper left')
plt.show()

# Train a Decission Tree

train_n = 0.6
train_size = math.floor(len(l) * train_n)

train_features = X[:train_size]
train_labels = y[:train_size]

test_features = X[train_size:]
test_labels = y[train_size:]

dt = tree.DecisionTreeClassifier()   
y_pred = dt.fit(train_features, train_labels).predict(test_features)   

# Query the accuracy of the model with the test set
acc = 100 * dt.score(test_features, test_labels)

# single-validation
print("Decision Tree Single-validation")
print("Number of mislabeled points out of a total %d points : %d"
    % (test_features.shape[0],(test_labels != y_pred).sum()))
print("Accuracy: %d pct"
    % (acc))

with open("Wholesale.dot", 'w') as f:
    f = tree.export_graphviz(dt, out_file=f)
    
# Eval a Naive Bayes Classifier
array = range(10, 90) # pct of the training set over the total dataset
pack =[]
for i in array:
    accuracy = []
        
    train_n = float(i) / 100
    
    train_size = math.floor(len(l) * train_n)
        
    train_features = X[:train_size]
    train_labels = y[:train_size]
        
    test_features = X[train_size:]
    test_labels = y[train_size:]    
        
    nb = GaussianNB() 
    y_pred = nb.fit(train_features, train_labels).predict(test_features)
    acc = 100 * nb.score(test_features, test_labels)
    
    # Query the accuracy of the model with the test set
    accuracy.append(acc)
        
    pack.append(accuracy)  
    
P = numpy.array(pack)

plt.xlabel('% of training set')
plt.ylabel('Accuracy')
plt.plot(array, P, label="NB Classifier")
plt.legend(loc='upper left')
plt.show()

# Train a Gaussian Naive Bayes Classifier
train_n = 0.6
train_size = math.floor(len(l) * train_n)

train_features = X[:train_size]
train_labels = y[:train_size]

test_features = X[train_size:]
test_labels = y[train_size:]

nb = GaussianNB()
y_pred = nb.fit(train_features, train_labels).predict(test_features)

# Query the accuracy of the model with the test set
acc = 100 * nb.score(test_features, test_labels)

# single-validation
print("NB Single-validation")
print("Number of mislabeled points out of a total %d points : %d"
    % (test_features.shape[0],(test_labels != y_pred).sum()))
print("Accuracy: %d pct"
    % (acc))    
    
# Train a SVM Clasifier
train_n = 0.6
train_size = math.floor(len(l) * train_n)

train_features = X[:train_size]
train_labels = y[:train_size]

test_features = X[train_size:]
test_labels = y[train_size:]  

linear_svc = svm.SVC(kernel="linear")
y_pred = linear_svc.fit(train_features, train_labels).predict(test_features)
acc = 100 * linear_svc.score(test_features, test_labels)
# single-validation
print("SVM with linear kernel Single-validation")
print("Number of mislabeled points out of a total %d points : %d"
    % (test_features.shape[0],(test_labels != y_pred).sum()))
print("Accuracy: %d pct"
    % (acc))    

rbf_svc = svm.SVC(kernel="rbf")
y_pred = rbf_svc.fit(train_features, train_labels).predict(test_features)
acc = 100 * rbf_svc.score(test_features, test_labels)
# single-validation
print("SVM with radial kernel Single-validation")
print("Number of mislabeled points out of a total %d points : %d"
    % (test_features.shape[0],(test_labels != y_pred).sum()))
print("Accuracy: %d pct"
    % (acc))    

poly_svc = svm.SVC(kernel="poly")
y_pred = poly_svc.fit(train_features, train_labels).predict(test_features)
acc = 100 * poly_svc.score(test_features, test_labels)
# single-validation
print("SVM with Polynomic kernel Single-validation")
print("Number of mislabeled points out of a total %d points : %d"
    % (test_features.shape[0],(test_labels != y_pred).sum()))
print("Accuracy: %d pct"
    % (acc))    



   