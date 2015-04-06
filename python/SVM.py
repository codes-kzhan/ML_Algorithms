import random
from numpy import *
import pandas as pd

def loadData(data, y):
    dataMat = []
    
    for t, row in enumerate(data):
        fltRow = map(float, row)
        dataMat.append(fltRow)
        
    dataMat = mat(dataMat)
    y0 = y[0]
    
    for t, i in enumerate(y):
        
        if i == y0:
            y[t] = 1
        else:
            
            y[t] = -1
            
    y = mat(y).transpose()
    return dataMat, y

def selectJrand(i, m):
    j = i
    
    while (j == i):
        j = int(random.uniform(0, m))
        
    return j

def clipAlpha(aj, H, L):
    
    if aj > H:
        aj = H
        
    if L > aj:
        aj = L
        
    return aj

                
class svm(object):
    
    def __init__(self, data, y, C = 1.0, tol = 0.001, \
                 maxIter = 40, ktup = ('lin', 0)):
        
        data, y = loadData(data, y)
        self.X = data
        self.y = y
        self.C = C
        self.tol = tol
        self.maxIter = maxIter
        self.ktup = ktup
        self.m, self.n = shape(data)
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.w = zeros((self.n, 1))
        self.eCache = mat(zeros((self.m, 2)))

    def calcEk(self, k):
        fXk = float(multiply(self.alphas, self.y).T * \
                    self.X * self.X[k].T) + self.b
        Ek = fXk - float(self.y[k])
        return Ek

    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]
        validEcacheList = nonzero(self.eCache[:,0].A)[0]
        
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        
        else:
            j = selectJrand(i, self.m)
            Ej = self.calcEk(j)
            return j, Ej

    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def innerL(self, i):
        y = self.y
        X = self.X
        alphas = self.alphas
        b = self.b
        tol = self.tol
        C = self.C
        
        Ei = self.calcEk(i)
        
        if ((y[i] * Ei < -tol) and (alphas[i] < C)) or \
            ((y[i] * Ei > tol) and (alphas[i] > 0)):
            
            j, Ej = self.selectJ(i, Ei)
            alphaIold = alphas[i].copy()
            alphaJold = alphas[j].copy()
            
            if (y[i] != y[j]):
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])
                
            else:
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
                
            if L == H:
                return 0
            
            eta = 2.0 * X[i] * X[j].T - X[i] * X[i].T - X[j] * X[j].T
            
            if eta >= 0:
                return 0
            
            alphas[j] -= y[j] * (Ei - Ej)/eta
            alphas[j] = clipAlpha(alphas[j], H, L)
            self.updateEk(j)
            
            if (abs(alphas[j] - alphaJold) < 0.00001):
                return 0
            
            alphas[i] += y[j] * y[i] * (alphaJold - alphas[j])
            self.updateEk(i)
            
            b1 = b - Ei - y[i] * (alphas[i] - alphaIold) * \
                 X[i] * X[i].T - y[j] * (alphas[j] - alphaJold) * \
                 X[i] * X[j].T
            
            b2 = b - Ej - y[i] * (alphas[i] - alphaIold) * \
                 X[i] * X[j].T - y[j] * (alphas[j] - alphaJold) * \
                 X[j] * X[j].T
            
            if (alphas[i] > 0) and (alphas[i] < C):
                b = b1
                
            elif (alphas[j] > 0) and (alphas[j] < C):
                b = b2
                
            else:
                b = (b1 + b2) / 2.0

            self.b = b
            return 1
        
        else:
            return 0

    def train(self):
        X = self.X
        y = self.y
        C = self.C
        tol = self.tol
        maxIter = self.maxIter
        ktup = self.ktup
        m = self.m
        alphas = self.alphas

        iterNum = 0
        entireSet = True
        alphaPairsChanged = 0
        
        while (iterNum < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):

            alphaPairsChanged = 0
            
            if entireSet:
                
                for i in range(m):
                    alphaPairsChanged += self.innerL(i)
                print "fullSet, iter: %d    pairs changed: %d" %\
                      (iterNum, alphaPairsChanged)
                    
                iterNum += 1
                
            else:
                nonBoundIs = nonzero((alphas.A > 0) * (alphas.A < C))[0]
                
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i)
                print "non-bound, iter: %d    pairs changed: %d" %\
                      (iterNum, alphaPairsChanged)
                    
                iterNum += 1
                
            if entireSet:
                entireSet = False
                
            elif (alphaPairsChanged == 0):
                entireSet = True
                
            print "iteration number: %d" % iterNum

    def predict(self, test):
        X = self.X
        y = self.y
        m = self.m
        alphas = self.alphas
        w = self.w
        b = self.b

        for i in range(m):
            w += multiply(alphas[i] * y[i], X[i,:].T)
            
        ret = []
        for i in test:
            calc = i * mat(w) + b
            if calc > 0:
                ret.append(1)
            else:
                ret.append(-1)
        return ret

            
            
if __name__ == '__main__':
    filename = "/home/yejiming/桌面/Kaggle/OttoGroup/train.csv"
    df = pd.read_csv(filename, header = 0)
    df = df.drop(['id'], axis = 1)
    data = df.values
    data = data[10000:26000]
    random.shuffle(data)
    y = data[0:100, -1]
    data = data[0:100, 0:-1]
    
    learner = svm(data, y)
    learner.train()
    yp = learner.predict(data)
