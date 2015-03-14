from csv import DictReader
from math import sqrt, fabs, exp, log
import numpy as np

D = 2 ** 20

# Neural Network withi a single hidden layer online learner
class NN(object):

    def __init__(self, n=D, h=50, alpha=0.1, l2=0., seed=0):

        """Initialize the NN class object.
        Args:
            n (int): number of input units
            h (int): number of hidden units
            a (double): initial learning rate
            l1 (double): L1 regularization parameter
            l2 (double): L2 regularization parameter
            seed (unsigned int): random seed
            interaction (boolean): whether to use 2nd order interaction or not
        """
        
        rng = np.random.RandomState(seed)

        self.n = n
        self.h = h
        self.alpha = alpha
        self.l2 = l2
        
        # weights between the hidden and output layers
        self.w1 = (rng.rand(h + 1) - .5) * 1e-7
        
        # weights between the input and hidden layers
        self.w0 = (rng.rand((n + 1) * h) - .5) * 1e-7

        # hidden units in the hidden layer
        self.z = np.zeros((h,), dtype=np.float64)

        # counters for biases and inputs
        self.c = 0.
        self.c1 = np.zeros((h,), dtype=np.float64)
        self.c0 = np.zeros((n,), dtype=np.float64)

    def predict(self, x):
        
        """Predict for features.
        Args:
            x : a list of value of non-zero features
        Returns:
            p (double): a prediction for input features
        """

        w0 = self.w0
        w1 = self.w1
        n = self.n
        h = self.h
        z = self.z
        
        # starting with the bias in the hidden layer
        p = w1[h]

        # calculating and adding values of hidden units
        for j in range(h):
            # starting with the bias in the input layer
            z[j] = w0[n * h + j]

            # calculating and adding values of input units
            for i in x:
                z[j] += w0[i * h + j]

            # apply the ReLU activation function to the hidden unit
            z[j] = z[j] if z[j] > 0. else 0.

            p += w1[j] * z[j]
        

        # apply the sigmoid activation function to the output unit
        return 1. / (1. + exp(-max(min(p, 35.), -35.)))

    def update(self, x, p, y):
        
        """Update the model.
        Args:
            x : a list of value of non-zero features
            p : predicted output
            y : target output
        Returns:
            updated model weights and counters
        """

        alpha = self.alpha
        l2 = self.l2
        n = self.n
        h = self.h
        w0 = self.w0
        w1 = self.w1
        c = self.c
        c0 = self.c0
        c1 = self.c1
        z = self.z
        
        e = p - y
        abs_e = fabs(e)
        dl_dy = e * alpha # dl/dy * (learning rate)

        # starting with the bias in the hidden layer
        w1[h] -= dl_dy / (sqrt(c) + 1) + l2 * w1[h]
        for j in range(h):
            # update weights related to non-zero hidden units
            if z[j] == 0.:
                continue

            # update weights between the hidden units and output
            # dl/dw1 = dl/dy * dy/dw1 = dl/dy * z
            w1[j] -= (dl_dy / (sqrt(c1[j]) + 1) * z[j] + l2 * w1[j])

            # starting with the bias in the input layer
            # dl/dz = dl/dy * dy/dz = dl/dy * w1
            dl_dz = dl_dy * w1[j]
            w0[n * h + j] -= (dl_dz / (sqrt(c1[j]) + 1) + l2 * w0[n * h + j])

            # update weights related to non-zero input units
            for i in x:
                # update weights between the hidden unit j and input i
                # dl/dw0 = dl/dz * dz/dw0 = dl/dz * v
                w0[i * h + j] -= (dl_dz / (sqrt(c0[i]) + 1) + l2 * w0[i * h + j])

                # update counter for the input i
                c0[i] += abs_e

            # update counter for the hidden unit j
            c1[j] += abs_e

        # update overall counter
        c += abs_e
