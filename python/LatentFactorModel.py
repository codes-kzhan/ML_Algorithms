import numpy as np
from math import exp, log, sqrt, fabs
import math
import random

filename = 'ratings.train.txt'
rng = np.random.RandomState(0)
k = 20
epoch = 40

def create_random_matrix(rows, cols):
    tmp = np.array([random.gauss(0, math.sqrt(5.0/cols)) for i in xrange(rows*cols)])
    tmp.shape = (rows,cols)
    return tmp

class LatentFeaturesModel(object):

    def __init__(self, alpha, l):
        # parameters
        self.alpha = alpha
        self.l = l
        self.Q = create_random_matrix(1682, k)
        self.P = create_random_matrix(943, k)
        
    def update(self, row):
        Q = self.Q
        P = self.P
        alpha = self.alpha
        l = self.l
        
        x = int(row[0]) - 1
        i = int(row[1]) - 1
        r = float(row[2])
        e = r - np.dot(Q[i], P[x])
        
        #Update q
        tmp_q = Q[i,:] + alpha*(e*P[x,:] - l*Q[i,:])
        
        #Update p
        tmp_p = P[x,:] + alpha*(e*Q[i,:] - l*P[x,:])

        Q[i,:] = tmp_q
        P[x,:] = tmp_p

def data(filename):
    for t, row in enumerate(open(filename)):
        row = row.strip().split()
        row[0] = int(row[0])
        row[1] = int(row[1])
        row[2] = int(row[2])
        yield t, row

def predict(learner):
    error = 0
    for t, row in enumerate(open(filename)):
        row = row.strip().split()
        x = int(row[0]) - 1
        i = int(row[1]) - 1
        r = float(row[2])
        error += math.pow((r - np.dot(learner.Q[i,:], learner.P[x,:])), 2)
    return error

def solution(learner):
    for i in range(epoch):
        for t, row in data(filename):
            learner.update(row)
        print predict(learner)

if __name__ == '__main__':
    learner = LatentFeaturesModel(alpha = 0.03, l = 0.2)
    solution(learner)
