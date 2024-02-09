# -*- coding: utf-8 -*-  
""" 
@author: Juan Águila Martínez (UOC - 2015) 
"""  

import numpy
from matplotlib import pyplot as plt

from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators
from pyevolve import Crossovers
from pyevolve import DBAdapters

import random

# Activity One: data analysis
packages = [20, 40, 25, 10, 30, 45, 80, 120, 110, 70, 85, 35, 60, 100, 90, 130]
# Length
print len(packages)
# Sort the list
print sorted(packages)
# Sum
print sum(packages)
# Mean
print sum(packages) / len(packages)

# Target function: maximize the weight in the first 5 containers
def eval_func(ind):
   score = 0.0
   package = [0 for x in range(6)] # 5 containers + the non included weights
   i = 0
   for x in ind:
       if (package[i] + x <= 200 or i == 5):
           package[i] = package[i] + x
       else:
           i = i + 1
           package[i] = package[i] + x
   for j in range(5):
       score = score + package[j]
   return score

packages = [20, 40, 25, 10, 30, 45, 80, 120, 110, 70, 85, 35, 60, 100, 90, 130]

def nonRepeatInitializer(genome, **args):
    genome.clearList()
    random.shuffle(packages)
    [genome.append(i) for i in packages]

# Genome instance
genome = G1DList.G1DList(len(packages))

# Change the initializator to custom values
genome.initializator.set(nonRepeatInitializer)

# Change the mutator to SWAP Mutator
genome.mutator.set(Mutators.G1DListMutatorSwap)

# The evaluator function (objective function)
genome.evaluator.set(eval_func)

# Change crossover to EDGE crossover
genome.crossover.set(Crossovers.G1DListCrossoverEdge)

# Best raw score
genome.setParams(bestrawscore=1000)

# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)
ga.selector.set(Selectors.GRouletteWheel)
ga.terminationCriteria.set(GSimpleGA.RawScoreCriteria)
# ga.nGenerations = 500

# Run the database
csv_adapter = DBAdapters.DBFileCSV(filename="pyEvolve.csv", identify="01",
                        frequency = 1, reset = True)
ga.setDBAdapter(csv_adapter)

k = 10

ga.evolve(k)

data=numpy.genfromtxt('pyEvolve.csv',delimiter=';')
generation = data[:,1]

maxData = data[:,9]

plt.plot(generation,maxData,marker='o',alpha=.5,label='Weight')
plt.title('50ind - 500it')
plt.xlabel('Generation')
plt.ylabel('Max Fitness Raw')
plt.ylim(900, 1000)







