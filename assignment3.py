# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:40:08 2019

@author: paulv

From: https://en.wikipedia.org/wiki/Random_sample_consensus?fbclid=IwAR1iVZJslckw_P_1W28RMlFvu9BE5HobQ4q_rntQhDX-5Aa__HVEUhITY58

Given:
    data – a set of observations
    model – a model to explain observed data points
    n – minimum number of data points required to estimate model parameters
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine data points that are fit well by model 
    d – number of close data points required to assert that a model fits well to data

Return:
    bestFit – model parameters which best fit the data (or nul if no good model is found)

iterations = 0
bestFit = nul
bestErr = something really large
while iterations < k {
    maybeInliers = n randomly selected values from data
    maybeModel = model parameters fitted to maybeInliers
    alsoInliers = empty set
    for every point in data not in maybeInliers {
        if point fits maybeModel with an error smaller than t
             add point to alsoInliers
    }
    if the number of elements in alsoInliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        betterModel = model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr = a measure of how well betterModel fits these points
        if thisErr < bestErr {
            bestFit = betterModel
            bestErr = thisErr
        }
    }
    increment iterations
}
return bestFit
"""

from sklearn.base import clone
from sklearn.metrics import mean_squared_error

# RANSAC algorithm
def RANSAC(points, model, n, k, t, d):
    iterations = 0
    bestFit = None
    bestErr = 1048576
    
    while iterations < k:
        maybeInliers = points[np.random.choice(points.shape[0], n, replace=False), :]
        maybeModel = clone(model)
        maybeModel.fit(maybeInliers[:,0].reshape(-1,1), maybeInliers[:,1])
        alsoInliers = np.zeros((0,2))
        
        for p in [p for p in points if p not in maybeInliers]:
            true, pred = p[1], maybeModel.predict(p[0].reshape(1,-1))
            if (abs(pred - true) < t):
                alsoInliers = np.append(alsoInliers, [p], axis=0)
        
        if len(alsoInliers) > d:
            union = np.append(maybeInliers, alsoInliers, axis=0)
            betterModel = maybeModel.fit(union[:,0].reshape(-1,1), union[:,1])
            
            true, pred = union[:,1], betterModel.predict(union[:,0].reshape(-1,1))
            thisErr = mean_squared_error(true, pred)
            if thisErr < bestErr:
                bestFit = betterModel
                bestErr = thisErr
        
        iterations += 1
    return bestFit

# Driver Code
min_x = 0
max_x = 10
points_in_model = 20
outliers_in_model = 5

# Creates the data points
import numpy as np
X = np.linspace(min_x, max_x, points_in_model, endpoint=True)
y = np.zeros(points_in_model + outliers_in_model)

# Makes the data realistic
import random
random.seed(1024)

# Initializes the y values of the points
for i in range(0, points_in_model):
    y[i] = X[i] + random.random()

# Initializes the x values of the outliers
for i in range(0, outliers_in_model):
    X = np.append(X, random.randint(min_x, max_x))

# Initializes the y values of the outliers
for i in range(points_in_model, points_in_model + outliers_in_model):
    y[i] = random.randint(min_x, max_x)

# Plots the data points
import matplotlib.pyplot as plt
plt.title('Data points')
plt.scatter(X, y, color='blue', linewidth=3)
plt.show()

# Models the data
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X.reshape(-1, 1), y)
prediction = regression.predict(X.reshape(-1, 1))

# Plots the data points with the model
import matplotlib.pyplot as plt
plt.title('Model without RANSAC')
plt.scatter(X, y, color='blue', linewidth=3)
plt.plot(X, prediction, color='red', linewidth=3)
plt.show()

# Combines points into a 2d array
data = np.append(X.reshape(-1, 1), y.reshape(-1, 1), axis=1)
ransac = RANSAC(data, LinearRegression(), 10, 200, 2, 5)
ransac_prediction = ransac.predict(X.reshape(-1, 1))

# Plots the data points with the model
import matplotlib.pyplot as plt
plt.title('Model with RANSAC')
plt.scatter(X, y, color='blue', linewidth=3)
plt.plot(X, ransac_prediction, color='red', linewidth=3)
plt.show()
