#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UOC Advanced Artificial Intelligence - 2014-2015 Q2
This code is a PCA Analysis based on the "Wholesale Customers" data
(Second deliverable of the course)
"""
# ------------------------------------------------------------------------------------------------
# Activity One: Load files and scale data
# ------------------------------------------------------------------------------------------------

import FunctionsForActivityOne as f # Import all needed functions

from numpy import *

from math import sqrt

import pylab as py

from sklearn.decomposition import PCA
# First, we read the data in Wholesale customers.csv
ratings = f.readRatings("Wholesale customers.csv")
# Now, we scale the distribution of every variable
scaledRatings = f.scaleRatings(ratings)
# The last step is to write the data
msg = f.writeStRatings(scaledRatings, "newWSC.data")  
print(msg)

def readFile(filename="input.txt"):
    # Read the csv file, ignoring first row
    lines = list(map(lambda l: [float(x) for x in (l.strip()).split("\t")], (open(filename, 'r').readlines())))
    return lines
    
do = readFile("newWSC.data")

for x in do:
    data.append(x[2:]) # Data

X = numpy.array(data)

# Apply PCA requesting all components (no argument)
mypca = PCA()
mypca.fit(X)
    
# How many components are required to explain 95% of the variance
acumvar = [sum(mypca.explained_variance_ratio_[:i+1]) for i in range(len(mypca.explained_variance_ratio_))]
print(list(zip(range(len(acumvar)), acumvar)))

pylab.plot(mypca.explained_variance_ratio_,'o-')
pylab.show()

# We will repeat the procedure with the non-scaled values
    
do2 = f.readRatings("Wholesale customers.csv")
# Apply PCA requesting all components (no argument)
for x in do2:
    data.append(x[2:])

X = numpy.array(data)
mypca = PCA()
mypca.fit(X)
    
# How many components are required to explain 95% of the variance
acumvar = [sum(mypca.explained_variance_ratio_[:i+1]) for i in range(len(mypca.explained_variance_ratio_))]
print(list(zip(range(len(acumvar)), acumvar)))

pylab.plot(mypca.explained_variance_ratio_,'o-')
pylab.show()

# ------------------------------------------------------------------------------------------------
# Activity Two + Three: Load previous file and solve the PCA Analysis
# ------------------------------------------------------------------------------------------------

# Split thelist in Labels + Data
data = []
target = []
# Comment and uncomment the labels to get the charts
# target_names = ['Hotel/Restaurant/Cafe ', 'Retail']
target_names = ['Lisbon', 'Oporto', 'Others']
target_names = numpy.array(target_names)

for x in do:
    data.append(x[2:])
    target.append(x[1]) # Switch the target names to show tendencies x[0]<-->x[1]

X = numpy.array(data)
y = numpy.array(target)

# mypca = PCA()
mypca = PCA(n_components=2)
X_r = mypca.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
     % str(mypca.explained_variance_ratio_))

plt.figure()
for c, i, target_name in zip("rg", [1, 2], target_names): # Switch the target [1,2]<-->x[1,2,3]
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of WholeSale Customers')
# Additionally, we will get the covariance     
cov = pca.get_covariance()
print(cov)

# ------------------------------------------------------------------------------------------------
# Activity Four: Solve MDS Analysis
# ------------------------------------------------------------------------------------------------

from matplotlib import pyplot as plt

from sklearn import manifold
from sklearn.metrics import euclidean_distances

similarities = euclidean_distances(X)

mds = manifold.MDS(n_components=2, dissimilarity="precomputed")
results = mds.fit(similarities)
coords = results.embedding_
plt.figure()
for c, i, target_name in zip("rg", [1, 2, 3], target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)
    # plt.scatter(X_r[y == i, 1], X_r[y == i, 2], c=c, label=target_name)
plt.legend()
plt.title('MDS of WholeSale Customers')
