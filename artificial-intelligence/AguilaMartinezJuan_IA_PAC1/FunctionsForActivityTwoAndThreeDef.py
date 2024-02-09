# -*- coding: utf-8 -*-

# ************************************************************
#               Activity Two and Three Functions
# ************************************************************

import FunctionsForActivityOneDef as f1

from random import sample
from itertools import repeat
from math import sqrt

def readRatingsDictio(file="outfile.data"):
    lines = f1.readRatings(file)
    dictio = {int(l[0]) : {}  for l in lines}
    for l in lines:
        # l[0] is the web id, l[1] is the user id, 
        # l[2..6] are the user's values for each topic.
        # There are an empty field (7) in each register... Is there a mistake?
        valuations = l[2:7]
        dictio[int(l[0])][int(l[1])] = valuations
    return dictio

# Compute the mean of each web to get a list of 5 valuations for each one
def meanValuations(dictio):
    dictioMeans = {}
    floatDictio = {}
    for web in dictio:
        floatDictio[web] = {}
        for user in dictio[web]:
            floatDictio[web][user] = []
            for value in dictio[web][user]:
                floatDictio[web][user].append(float(value))
    for web in floatDictio:
        # Each web has a dictio {user: [5 valuations]}
        vals = zip(*floatDictio[web].values())
        dictioMeans[web] = list(map(lambda x:sum(x)/len(x), vals))
    return dictioMeans
    
def euclideanDist(list1, list2):
    # Compute the sum of squares of the two lists (should have same length)
    sum2 = sum([pow(list1[i]-list2[i], 2) for i in range(len(list1))])
    return sqrt(sum2)

def euclideanSimilarity(list1, list2):
    return 1/(1+euclideanDist(list1, list2)) 
   
# Given a dictionary like {key1 : [values]} it computes k-means
# clustering, with k groups, executing maxit iterations at most, using
# the specified similarity function.
# It returns two things (as a tuple):
# -{key1:cluster number} with the cluster assignemnts (which cluster
#  does each element belong to
# -[{key2:values}] a list with the k centroids (means of the values
#  for each cluster.
# All values should have the same length, and are interpreted as coordinates
# of the elements to be clustered.
def kmeans_list(dictionary, k, maxit, similarity = euclideanSimilarity):
    # First k random points are taken as initial centroids.
    # Each centroid is [values]
    centroids = [dictionary[x] for x in sample(dictionary.keys(), k)]
    # Assign each key1 to a cluster number 
    previous   = {}
    assignment = {}
    # On each iteration it assigns points to the centroids and computes
    # new centroids
    for it in range(maxit):
        # Assign points to the closest centroids
        for key1 in dictionary:
            simils = map(similarity,repeat(dictionary[key1],k), centroids)
            assignment[key1] = simils.index(max(simils))           
        # If there are no changes in the assignment then finish
        if previous == assignment:
            break
        previous.update(assignment)
        # Recompute centroids: annotate coords of points in each cluster
        # like {idcluster: [[values1], [values2], ...]
        coords   = {x : [] for x in range(k)}
        for key1 in dictionary:
            group = assignment[key1]
            coords[group].append(dictionary[key1])
        # Compute means (new centroids)
        centroids = []
        for group in coords:
            vals = zip(*coords[group])
            centroids.append(list(map(lambda x:sum(x)/len(x), vals)))
        if None in centroids: break
    return (assignment, centroids)





