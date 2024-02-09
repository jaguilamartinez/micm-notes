# -*- coding: utf-8 -*-

import FunctionsForActivityFourDef as f4

# We will use the scaled values for this activity
valuations = readUserValuations("newScWebs.data")
favorites = readFavorites("favorits.data")

# Run the recommender. We get {user: (list of recommended webs in desc order)}
recomPearson = {usr : zip(*weightedRating(valuations, usr, pearsonCoeff_list))[0] for usr in valuations.keys() }

# print(recomPearson)
             
# Finally compute the average position of the favorite hotel in the 
# recommendation for each user. The closest to position 0, the better the
# recommender has performed
positions = [recomPearson[usr].index(favorites[usr]) for usr in favorites]                    

meanPosition = sum(positions)/float(len(positions))

print(meanPosition)

