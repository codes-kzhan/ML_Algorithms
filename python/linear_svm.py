import numpy as np
import pandas as pd
from math import exp, log, sqrt, fabs
import math
import random

train_df = pd.read_csv(r'features.train.txt', header = 1)
test_df = pd.read_csv(r'features.test.txt', header = 1)
y_df = pd.read_csv(r'target.train.txt', header = 1)
yt_df = pd.read_csv(r'target.test.txt', header = 1)

train_data = train_df.values
y = y_df.values
test_data = test_df.values
yt = yt_df.values

rng = np.random.RandomState(0)
e_batch = 0.25

class linear_svm(object):

    def __init__(self, alpha = 0.0001, C = 1, features = 122,
                 mode = 'mini_batch', echo = 50, batch_size = 20):
        # parameters
        self.alpha = alpha
        self.C = C
        self.w = (rng.rand(features) - .5)
        self.b = rng.rand() - .5
        self.k = 0
        self.features = features
        self.mode = mode
        self.echo = echo
        if mode == 'mini_batch':
            self.batch_size = batch_size
        elif batch_size != 20:
            print "Error: batch_size only available for mini_batch method."

    def loss(self, train_data, y):
        C = self.C
        w = self.w
        b = self.b

        cost = 0
        for i, x in enumerate(train_data):
            cost += C * max(0, 1 - y[i] * ((w * x).sum() + b))

        cost += 0.5 * (w ** 2).sum()
        self.k += 1
        
        return cost / len(y)

    def update_batch(self, train_data, y):
        features = self.features
        C = self.C
        alpha = self.alpha
        w = self.w
        b = self.b
        
        for j in range(features):
            gradient = 0
            for i, x in enumerate(train_data):
                if (y[i] * ((w * x).sum() + b) >= 1):
                    continue
                else:
                    gradient -= y[i] * x[j]
            w[j] -= alpha * (w[j] + C * gradient)
        
        gradient = 0
        for i, x in enumerate(train_data):
            if (y[i] * ((w * x).sum() + b) >= 1):
                continue
            else:
                gradient -= y[i]
        b -= alpha * (C * gradient)

    def update_sgd(self, train_data, y):
        features = self.features
        C = self.C
        alpha = self.alpha
        w = self.w
        b = self.b

        whole_data = np.hstack([train_data, y])
        np.random.shuffle(whole_data)
        
        for i, row in enumerate(whole_data):

            x = row[0 : -1]
            y = row[-1]
            
            for j in range(features):
                if (y * ((w * x).sum() + b) >= 1):
                    gradient = 0
                else:
                    gradient = -y * x[j]
                w[j] -= alpha * (w[j] + C * gradient)
                
            if (y * ((w * x).sum() + b) >= 1):
                gradient = 0
            else:
                gradient = -y
            b -= alpha * (C * gradient)

    def update_mini(self, train_data, y):
        features = self.features
        C = self.C
        alpha = self.alpha
        w = self.w
        b = self.b
        batch_size = self.batch_size

        whole_data = np.hstack([train_data, y])
        np.random.shuffle(whole_data)

        for i, row in enumerate(whole_data):

            x = row[0 : -1]
            y = row[-1]
            gradient = [0] * features
            g = 0

            if (y * ((w * x).sum() + b) >= 1):
                gradient = gradient
            else:
                for j in range(features):
                    gradient[j] -= y * x[j]
                g -= y
            
            if (i + 1) % batch_size == 0 or i == (len(whole_data) - 1):
                for j in range(features):
                    w[j] -= alpha * (w[j] + C * gradient[j])
                b -= alpha * (C * g)

        

    def train(self, train_data, y):
        echo = self.echo
        mode = self.mode
        
        if (mode == 'batch'):
            lastCost = 2.0
            improve = 1.0
            for i in range(echo):
                learner.update_batch(train_data, y)
                currentCost = learner.loss(train_data, y)
                improve = lastCost - currentCost
                lastCost = currentCost
                print "echo: " + str(learner.k) + "     ",
                print "Current Cost: " + str(currentCost[0]) + \
                "       " + "Improvement: " + str(improve[0])

        if (mode == 'sgd'):
            lastCost = 2.0
            improve = 1.0
            for i in range(echo):
                learner.update_sgd(train_data, y)
                currentCost = learner.loss(train_data, y)
                improve = lastCost - currentCost
                lastCost = currentCost
                print "echo: " + str(learner.k) + "     ",
                print "Current Cost: " + str(currentCost[0]) + \
                "       " + "Improvement: " + str(improve[0])

        if (mode == 'mini_batch'):
            lastCost = 2.0
            improve = 1.0
            for i in range(echo):
                learner.update_mini(train_data, y)
                currentCost = learner.loss(train_data, y)
                improve = lastCost - currentCost
                lastCost = currentCost
                print "echo: " + str(learner.k) + "     ",
                print "Current Cost: " + str(currentCost[0]) + \
                "       " + "Improvement: " + str(improve[0])

        
if __name__ == '__main__':
    learner = linear_svm()
    learner.train(train_data, y)
