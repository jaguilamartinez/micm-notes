# -*- coding: utf-8 -*-

# ************************************************************
#               Activity One Functions
# ************************************************************

from math import sqrt
import csv # CSV Read and Write Functions

def mean(lst):
    # Calculates mean for lst
    return sum(lst) / len(lst)

def stddev(lst):
    # Returns the standard deviation of lst
    mn = mean(lst)
    variance = sum([(e-mn)**2 for e in lst])
    return sqrt(variance)

def readRatings(filename="webs.data"):
    # Reads CSV to load a list of lists with every cell value
    lines = [(l.strip()).split("\t") for l in (open(filename).readlines())]
    return lines
    # It will be used in the next activity too
    
def standarizeRatings(array):
    # The result of standardization (or Z-score normalization) is that 
    # the features will be rescaled so that they'll have the properties of a 
    # standard normal distribution with mu = 0 and sigma = 1
    aLength = len(array[0])
    rawData = {x: [] for x in range(2, aLength)}
    means = []
    stDevs = []
    for l in array:
        for x in range(2, aLength):
            rawData[x].append(float(l[x]))
    for d in rawData:
        means.append(mean(rawData[d]))
        stDevs.append(stddev(rawData[d]))
    newArray = []
    for l in array:
        tmp = []
        tmp.append(int(l[0])) # Append idWeb
        tmp.append(int(l[1])) # Append idUser
        for x in range(2, aLength):
            if stDevs[x - 2] != 0:
                value = (float(l[x]) - means[x - 2]) / stDevs[x - 2]
                tmp.append(value)
            else:
                value = 0
                tmp.append(means[x - 2])
        newArray.append(tmp)
    return newArray
    
def writeStRatings(array, name="output.data"):
    # AWrite the data
    with open(name, "w") as fp:
        a = csv.writer(fp, delimiter="\t")
        a.writerows(array)
        msg = "Data succesfully written in file " + name
        return msg

# Vector with the maximum values for each topic
MAX_VALUATIONS = [30.0, 100.0, 50.0, 10.0, 10.0, 0]

def scaleVals(vals):
    # Auxiliary function that receives a list of valuations as read
    # from the file and returns it scaled to 0..1 
    return [(float(vals[i])-1)/(MAX_VALUATIONS[i]-1) for i in range(len(vals))]

def scaleRatings(array):
    newArray = []
    for l in array:
        tmp = []
        tmp.append(int(l[0])) # Append idWeb
        tmp.append(int(l[1])) # Append idUser
        values = scaleVals(l[2:])
        for x in values:
            tmp.append(x)
        newArray.append(tmp)
    return newArray
    
    
