from csv import DictReader
from math import exp, log, sqrt, fabs
import random
import numpy as np

train = open(r'pre_train.csv') # Input the training data
test = open(r'pre_test.csv')   # Input the testing data

alpha = .1
k = 10
D = 2 ** 22
rng = np.random.RandomState(0)

class fact_machine(object):
    """Factorization Machine online learner.
    
    Parameter:
    ----------
    n (int): number of features after hashing trick
    alpha (float): initial learning rate
    w0 (float): weight for bias
    k (int): size of factors for interactions
    c0 (float): counters
    w (list of float): feature weights
    c (list of float): counters for weights
    V (list of float): feature weights for factors
    D (long): maximum value for hash
    """

    def __init__(self, alpha, D):
         """Initialize the FM class object.
        Parameters:
        ----------
        alpha (double): initial learning rate
        D (long): maximum value for hash
        """
        self.alpha = alpha

        self.w0 = 0.
        self.c0 = 0.
        self.w = [0.] * D
        self.c = [0.] * D
        self.v = (rng.rand(D * k) - .5) * 1e-6

    def predict(self, x):
        """Predict for features.
        Parameters:
        ----------
        x (list of int): a list of index of non-zero features
        
        Outputs:
        ----------
        p (float): prediction for input features
        """
        # weights
        w0 = self.w0
        w = self.w
        v = self.v

        wx = 0.
        vx = np.zeros((k,), dtype=np.float64)
        v2x2 = np.zeros((k,), dtype=np.float64)
        

        for i in x:
            wx += w[i]
            for j in range(k):
                vx[j] += v[i * k + j]
                v2x2[j] += v[i * k + j] ** 2

        p = w0 + wx
        for i in range(k):
            p += 0.5 * (vx[i] ** 2 - v2x2[i])

        # bounded sigmoid function, this is the probability of being clicked
        return 1. / (1. + exp(-max(min(p, 35.), -35.)))

    def update(self, x, p, y):
        """Update the model.
        Parameters:
        ----------
        x (list of int): a list of index of non-zero features
        p (float): prediction for input features
        y (int): value of the target
        
        Outputs:
        ----------
        updates model weights and counts
        """
        # parameter
        alpha = self.alpha

        # model
        w0 = self.w0
        c0 = self.c0
        w = self.w
        v = self.v
        c = self.c

        vx = np.zeros((k,), dtype=np.float64)
        for i in x:
            for j in range(k):
                vx[j] += v[i * k + k]

        # gradient under logloss
        e = p - y
        abs_e = fabs(e)
        c[0] += abs_e
        
        # update w0
        w[0] -= alpha / (sqrt(c[0]) + 1) * e

        # update w
        for i in x:
            dl_dw = alpha / (sqrt(c[i]) + 1) * e
            w[i] -= dl_dw
            c[i] += abs_e
            for f in range(k):
                v[i * k + f] -= dl_dw * (vx[f] - v[i * k + f])
