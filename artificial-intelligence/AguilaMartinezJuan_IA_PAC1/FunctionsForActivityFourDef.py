# -*- coding: utf-8 -*-

# ************************************************************
#                   Activity Four Functions
# ************************************************************

from math import sqrt

# Reads webs.data and returns a dictionary of each web user's values as:
# {user: {web:[list of values]}}
def readUserValuations(filename="webs.data"):
    myfile = file(filename)
    lines = [(l.strip()).split("\t") for l in myfile.readlines()]
    # l[1] is the user id
    dictio = {int(l[1]) : {}  for l in lines}
    for l in lines:
        # l[0] is the web id, l[2..6] are the user's values.
        valuations = []
        for x in l[2:]:
            valuations.append(float(x))
        dictio[int(l[1])][int(l[0])] = valuations
    return dictio

# Read favorits.data file, return {user: favorite web}
def readFavorites(filename="favorits.data"):
    myfile = file(filename)
    lines = [(l.strip()).split("\t") for l in myfile.readlines()]
    # l[0] is the user id, l[1] the web id
    dictio = {int(l[0]) : int(l[1])  for l in lines}
    return dictio

# A simple Pearson correlation function between two lists
def simplePearson(list1, list2):
    mean1 = sum(list1)
    mean2 = sum(list2)
    num  = sum([(list1[i]-mean1)*(list2[i]*mean2) for i in range(len(list1))])    
    den1 = sqrt(sum([pow(list1[i]-mean1, 2) for i in range(len(list1))]))
    den2 = sqrt(sum([pow(list2[i]-mean2, 2) for i in range(len(list2))]))
    den  = den1*den2
    if den==0:
        return 0
    return num/den

# Compute the mean Pearson coeff between a pair of users
# Input two diccionaries with ratings of each user {web: [valuations]}
def pearsonCoeff_list(user1, user2):
    # Retrieve the webs common to both users
    commons  = [x for x in user1 if x in user2]
    nCommons = float(len(commons))
    # If there are no common elements, return zero; otherwise
    # compute the coefficient
    if nCommons==0:
        return 0
    # Compute the mean Pearson coeff for all common webs
    return sum([simplePearson(user1[x], user2[x]) for x in commons])/nCommons

def pearsonCoeff(dic1, dic2):
    # Retrieve the elements common to both dictionaries
    commons  = [x for x in dic1 if x in dic2]
    nCommons = float(len(commons))

    # If there are no common elements, return zero; otherwise
    # compute the coefficient
    if nCommons==0:
        return 0

    # Compute the means of each dictionary
    mean1 = sum([dic1[x] for x in commons])/nCommons
    mean2 = sum([dic2[x] for x in commons])/nCommons

    # Compute numerator and denominator
    num  = sum([(dic1[x]-mean1)*(dic2[x]-mean2) for x in commons])
    den1 = sqrt(sum([pow(dic1[x]-mean1, 2) for x in commons]))
    den2 = sqrt(sum([pow(dic2[x]-mean2, 2) for x in commons]))
    den  = den1*den2

    # Compute the coefficient if possible or return zero
    if den==0:
        return 0

    return num/den

# Produces a sorted list of weighted ratings from a dictionary of
# user ratings and a user id.
# You can choose the function of similarity between users.
def weightedRating(dictio, user, similarity = pearsonCoeff):
    # In the first place a dictionary is generated with the similarities
    # of our user with all other users.
    # This dictionary could be stored to avoid recomputing it.
    simils = {x: similarity(dictio[user], dictio[x])
              for x in dictio if x != user}

    # Auxiliary dictionaries {webId: [rating*users similarity]}
    # and {webId: [users similarity]} (numerator and denominator
    # of the weighted rating)
    numerator   = {}
    denominator = {}

    # The ratings dictionary is traversed, while filling the auxiliary
    # dictionaries with the values found.
    for userId in simils:
        for webId in dictio[userId]:
            if not numerator.has_key(webId):
                numerator  [webId] = []
                denominator[webId] = []
            s = simils[userId]
            # PAC1 2014: add the sum of valorations as there are many of them
            numerator  [webId].append(sum(dictio[userId][webId])*s)
            denominator[webId].append(s)

    # Compute and sort weighted ratings    
    result = []
    for webId in numerator:
        s1 = sum(numerator  [webId])
        s2 = sum(denominator[webId])
        if s2 == 0:
            mean = 0.0
        else:
            mean = s1/s2

	# Append the rating only if the user does not have it already
	if not dictio[user].has_key(webId):
	        result.append((webId,mean))

    result.sort(key = lambda x: x[1], reverse=True)
    return result