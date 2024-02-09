# -*- coding: utf-8 -*-
"""
@author: Juan Águila Martínez (UOC - 2015)
"""

import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from random import shuffle
import plotDecisionSurface

#####################################################################             
# Activitat 2
#####################################################################

# Generem dos conjunts de dades bidimensionals, utilitzant
# el mètode random multivariate normal
mean = [2,-4] # Vector de mitjanes
cov = [[2,-1],[-1,2]] # Matriu de covariància
A = np.random.multivariate_normal(mean,cov,2000)

mean = [1,-3]
cov = [[1,1.5],[1.5,3]] 
B = np.random.multivariate_normal(mean,cov,2000)

# Representem els punts en un scatter plot
plt.scatter(A[:,0],A[:,1],color = '#F5CA0C', marker='v', label="$1st Ds - A$")
plt.scatter(B[:,0],B[:,1],color = '#00A99D', marker='x', label="$2nd Ds - B$")
plt.title("Scatter plot of both datasets")
plt.legend()
plt.show()

# Afegim una etiqueta a l'array, per a avaluar la classificació
tmp = np.zeros((2000, 1))
tmp[:] = 1
A = np.append(A,tmp,1)
tmp[:] = 2
B = np.append(B,tmp,1)

shuffle(A)
shuffle(B)

# I generem els vectors definitius que utilitzarem
X_train = np.concatenate((A[:1000,:2], B[:1000,:2]), axis=0 )
y_train = np.concatenate((A[:1000,2:], B[:1000,2:]), axis=0 )
# Reshape
y_train = np.ravel(y_train)

X_test  = np.concatenate((A[1000:,:2], B[1000:,:2]), axis=0 )
y_test  = np.concatenate((A[1000:,2:], B[1000:,2:]), axis=0 )
# Reshape
y_test = np.ravel(y_test)

# Entrenem el classificador
nb = GaussianNB()
y_pred = nb.fit(X_train, y_train).predict(X_test)

# Avaluem la classificació
acc = 100 * nb.score(X_test, y_test)
print("NB Single-validation")
print("Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0],(y_test != y_pred).sum()))
print("Accuracy: %d pct"
    % (acc))    
plot_decision_surface(nb, X_test, y_test, 'Naive Bayes - Test data')    

# Entrenem el classificador
lda = LDA()
y_pred = lda.fit(X_train, y_train).predict(X_test)

# Avaluem la classificació
acc = 100 * lda.score(X_test, y_test)
print("LDA Single-validation")
print("Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0],(y_test != y_pred).sum()))
print("Accuracy: %d pct"
    % (acc))    
plot_decision_surface(lda, X_test, y_test, 'LDA - Test data')
    


