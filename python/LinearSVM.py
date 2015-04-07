import sys
import numpy as np
import pandas as pd
from math import exp, log, sqrt, fabs
import math
import random

rng = np.random.RandomState(0)

def loadData(data, y):
    dataMat = []
    
    for t, row in enumerate(data):
        fltRow = map(float, row)
        dataMat.append(fltRow)
        
    dataMat = np.array(dataMat)
    y0 = y[0]
    
    for t, i in enumerate(y):
        
        if i == y0:
            y[t] = 1
        else:
            
            y[t] = -1
            
    y = np.array(y).transpose()
    return dataMat, y


class linear_svm(object):

    def __init__(self, alpha = 0.0005, C = 1.0, n = 93,
                 mode = 'batch', echo = 100, batch_size = 20,
                 criteria = 'accuracy'):
        # parameters
        self.alpha = alpha
        self.C = C
        self.w = (rng.rand(n) - .5)
        self.b = rng.rand() - .5
        self.k = 0
        self.n = n
        self.mode = mode
        self.echo = echo
        self.criteria = criteria
        
        if mode == 'mini_batch':
            self.batch_size = batch_size
        elif batch_size != 20:
            print "Error: batch_size only available for mini_batch method."

    def loss(self, train_data, y):
        C = self.C
        w = self.w
        b = self.b
        criteria = self.criteria

        if criteria == 'loss':
            cost = 0
            for i, x in enumerate(train_data):
                cost += C * max(0, 1 - y[i] * ((w * x).sum() + b))

            cost += 0.5 * (w ** 2).sum()
            self.k += 1
            return cost

        if criteria == 'accuracy':
            cost = 0
            for i, x in enumerate(train_data):
                if (y[i] * ((w * x).sum() + b) >= 0):
                    continue
                else:
                    cost += 1
            self.k += 1
                    
        return float(cost) / len(y)

    def update_batch(self, train_data, y):
        n = self.n
        C = self.C
        alpha = self.alpha
        w = self.w
        b = self.b

        for j in range(n):
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
        n = self.n
        C = self.C
        alpha = self.alpha
        w = self.w
        b = self.b

        try:
            whole_data = np.hstack([train_data, y])
        except:
            rows = []
            for i in y:
                row = [i]
                rows.append(row)
            y = np.array(rows)
            whole_data = np.hstack([train_data, y])
            
        np.random.shuffle(whole_data)
        
        for i, row in enumerate(whole_data):

            x = row[0 : -1]
            y = row[-1]
            
            for j in range(n):
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
        n = self.n
        C = self.C
        alpha = self.alpha
        w = self.w
        b = self.b
        batch_size = self.batch_size

        try:
            whole_data = np.hstack([train_data, y])
        except:
            rows = []
            for i in y:
                row = [i]
                rows.append(row)
            y = np.array(rows)
            whole_data = np.hstack([train_data, y])
            
        np.random.shuffle(whole_data)

        for i, row in enumerate(whole_data):

            x = row[0 : -1]
            y = row[-1]
            gradient = [0] * n
            g = 0

            if (y * ((w * x).sum() + b) >= 1):
                gradient = gradient
            else:
                for j in range(n):
                    gradient[j] -= y * x[j]
                g -= y
            
            if (i + 1) % batch_size == 0 or i == (len(whole_data) - 1):
                for j in range(n):
                    w[j] -= alpha * (w[j] + C * gradient[j])
                b -= alpha * (C * g)

        

    def train(self, data, y):
        train_data, y = loadData(data, y)
        n = self.n
        echo = self.echo
        mode = self.mode

        if n != len(train_data[0]):
            print "Error: number of features not correct(should be " + \
                  str(len(train_data[0])) + ")"
            sys.exit()
            
        if (mode == 'batch'):
            lastCost = 2.0
            improve = 1.0
            for i in range(echo):
                self.update_batch(train_data, y)
                currentCost = self.loss(train_data, y)
                improve = lastCost - currentCost
                lastCost = currentCost
                print "echo: " + str(self.k) + "     ",
                print "Current Cost: " + str(currentCost) + \
                "       " + "Improvement: " + str(improve)

        if (mode == 'sgd'):
            lastCost = 2.0
            improve = 1.0
            for i in range(echo):
                self.update_sgd(train_data, y)
                currentCost = self.loss(train_data, y)
                improve = lastCost - currentCost
                lastCost = currentCost
                print "echo: " + str(self.k) + "     ",
                print "Current Cost: " + str(currentCost) + \
                "       " + "Improvement: " + str(improve)

        if (mode == 'mini_batch'):
            lastCost = 2.0
            improve = 1.0
            for i in range(echo):
                self.update_mini(train_data, y)
                currentCost = self.loss(train_data, y)
                improve = lastCost - currentCost
                lastCost = currentCost
                print "echo: " + str(self.k) + "     ",
                print "Current Cost: " + str(currentCost) + \
                "       " + "Improvement: " + str(improve)

    def predict(self, data):
        w = self.w
        b = self.b

        ret = []
        for x in data:
            calc = (w * x).sum() + b
            if calc >= 0:
                ret.append(1)
            else:
                ret.append(-1)
        return ret

    def score(self, test_data, ytest):
        test_data, ytest = loadData(test_data, ytest)
        w = self.w
        b = self.b

        ret = 0
        for i, x in enumerate(test_data):
            if (ytest[i] * ((w * x).sum() + b) >= 0):
                ret += 1
        return float(ret)/len(ytest)

        
if __name__ == '__main__':
    filename = "/home/yejiming/桌面/Kaggle/OttoGroup/train.csv"
    df = pd.read_csv(filename, header = 0)
    df = df.drop(['id'], axis = 1)
    data = df.values
    data = data[10000:26000]
    np.random.shuffle(data)
    y = data[0:200, -1]
    data = data[0:200, 0:-1]
    train = data[0:100]
    ytrain = y[0:100]
    test = data[100:200]
    ytest = y[100:200]
    learner = linear_svm()
    learner.train(train, ytrain)
    score = learner.score(test, ytest)
