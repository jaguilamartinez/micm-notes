#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Activity 2: First Cluster Analysis

import FunctionsForActivityTwo as f2

# Scaled Data Test

ratingsDictio = f2.readRatingsDictio("newScWebs.data")
# print(ratingsDictio)
means = f2.meanValuations(ratingsDictio)
# print(means)
(assignmentSc, centroidsSc) = f2.kmeans_list(means, 6, 20)
print(assignmentSc)

# Standarized Data Test

ratingsDictio = f2.readRatingsDictio("newStWebs.data")
# print(ratingsDictio)
means = f2.meanValuations(ratingsDictio)
# print(means)
(assignmentSt, centroidsSt) = f2.kmeans_list(means, 6, 20)
print(assignmentSt)

# Activity 3: Adjusted Rand Index
from sklearn import metrics

# Reference clustering
labels_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Scaled
print(metrics.adjusted_rand_score(labels_true, list(assignmentSc.values())))

# Standarized
print(metrics.adjusted_rand_score(labels_true, list(assignmentSt.values())))