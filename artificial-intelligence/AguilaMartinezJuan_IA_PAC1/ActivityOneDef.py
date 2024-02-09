#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Activity 1: Data Standarization

import FunctionsForActivityOne as f # Import all needed functions

# First, we read the data in webs.data
ratings = f.readRatings("webs.data")
# Now, we standarize the distribution of every variable
standarizedRatings = f.standarizeRatings(ratings)
# The last step is to write the data
msg = f.writeStRatings(standarizedRatings, "newStWebs.data")  
print(msg)
# Additionally, we will try with another scaling function to compare results
scaledRatings = f.scaleRatings(ratings)
msg = f.writeStRatings(scaledRatings, "newScWebs.data")  
print(msg)
