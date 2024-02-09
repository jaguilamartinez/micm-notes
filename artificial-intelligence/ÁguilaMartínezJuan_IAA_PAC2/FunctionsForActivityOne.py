# -*- coding: utf-8 -*-

# ************************************************************
#               Activity One Functions
# ************************************************************

from math import sqrt
import csv # CSV Read and Write Functions

def readRatings(filename="input.data"):
    with open(filename, 'rU') as f:
        reader = csv.reader(f,delimiter=';')
        content = list(reader)
    return content
    # It will be used in the next activity too
    
def writeStRatings(array, name="output.data"):
    # AWrite the data
    with open(name, "w") as fp:
        a = csv.writer(fp, delimiter="\t")
        a.writerows(array)
        msg = "Data succesfully written in file " + name
        return msg

# Vector with the maximum values for each topic
MAX_VALUATIONS = [112151, 73498, 92780, 60869, 40827, 47943]

def scaleVals(vals):
    # Auxiliary function that receives a list of valuations as read
    # from the file and returns it scaled to 0..1 
    return [(float(vals[i])-1)/(MAX_VALUATIONS[i]-1) for i in range(len(vals))]

def scaleRatings(array):
    aLength = len(array[0])
    newArray = []
    for l in array:
        tmp = []
        tmp.append(int(l[0])) # Append channel
        tmp.append(int(l[1])) # Append region
        values = scaleVals(l[2:])
        for x in values:
            tmp.append(x)
        newArray.append(tmp)
    return newArray
    
    
