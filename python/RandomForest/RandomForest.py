import numpy as np
import pandas as pd
from cart import *
import random

def loadData(data, y):
    dataMat = []
    
    for t, row in enumerate(data):
        fltRow = map(float, row)
        dataMat.append(fltRow)
        
    dataMat = np.mat(dataMat)
    y = np.mat(y).transpose()
    return dataMat, y

def labelEncoder(y):
    yDict = {}
    
    for i in y:
        yDict.setdefault(i, 0)
        yDict[i] += 1

    yList = [key for key in yDict]
    yList.sort()
    
    for t, item in enumerate(y):
        y[t] = yList.index(item)

    return np.array(y)
    
def bootstrap(data, y):
    ## Make bootstrap
    number = len(y)
    dataTrain = []
    yTrain = []
    
    for i in range(number):
        randNum = random.randint(0,len(data) - 1)
        dataTrain.append(data[randNum])
        yTrain.append(y[randNum])

    yTest = y
    dataTest = data

    ## Transform dataset to numpy arrays
    dataTrain = np.array(dataTrain)
    dataTest = np.array(dataTest)
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)
    return yTrain, dataTrain, yTest, dataTest

def argmax(dictionary):
    maximum = 0
    
    for key in dictionary:
        if dictionary[key] > maximum:
            maximum = dictionary[key]
            maxId = key
            
    return maxId

def correctNum(y, y_pred):
    ret = 0
    
    for t, i in enumerate(y):
        if i == y_pred[t]:
            ret += 1
            
    return ret

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
    
class RandomForest(object):

    def __init__(self, n_estimators=100, max_depth=inf, max_features=10,
                 criteria='logloss'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criteria = criteria
        self.forest = []

    def validation(self, yTest, dataTest, forest):
        retProba = [{} for i in range(len(yTest))]
        
        for tree in forest:
            probas = tree.predict_proba(dataTest)
            for t, p in enumerate(probas):
                for key in p:
                    retProba[t].setdefault(key, 0)
                    retProba[t][key] += p[key] / len(forest)

        yList = list(set(yTest))
        yList.sort()
        Proba = []
        
        for p in retProba:
            Ap = []
            for i in yList:
                if i not in p:
                    Ap.append(0.0)
                else:
                    Ap.append(p[i])
            Proba.append(Ap)
            
        Proba = np.array(Proba)
        retClass = []
        
        for i in retProba:
            maxId = argmax(i)
            retClass.append(maxId)

        ret = correctNum(yTest, retClass)
        retP = multiclass_log_loss(yTest, Proba)
        return retP, 1 - (float(ret)/len(yTest))

    def train(self, data, y, test, yt):
        n_estimators = self.n_estimators
        max_depth = self.max_depth
        max_features = self.max_features
        criteria = self.criteria
        forest = self.forest

        for i in range(n_estimators):
            yTrain, dataTrain, yTest, dataTest = bootstrap(data, y)
            tree = ClassificationTree(maxDepth = max_depth, maxFeatures = max_features)
            print "iteration: ",
            tree.train(dataTrain, yTrain)
            print i + 1,
            forest.append(tree)
            
            if criteria == None:
                continue
            
            loglossOOB, errorOOB = self.validation(yTest, dataTest, forest)
            loglossTest, errorTest = self.validation(yt, test, forest)
            
            if criteria == 'error':
                print "       OOB error: %0.5f" % errorOOB,
                print "       test set error: %0.5f" % errorTest
                
            elif criteria == 'logloss':
                print "       OOB log loss:: %0.5f" % loglossOOB,
                print "       test set log loss: %0.5f" % loglossTest
            

    def predict(self, data):
        forest = self.forest
        retProba = [{} for i in range(len(data))]
        
        for tree in forest:
            probas = tree.predict_proba(data)
            for t, p in enumerate(probas):
                for key in p:
                    retProba[t].setdefault(key, 0)
                    retProba[t][key] += p[key]

        retClass = []
        for i in retProba:
            maxId = argmax(i)
            retClass.append(maxId)
            
        return np.array(retClass)

    def predict_proba(self, data):
        forest = self.forest
        retProba = [{} for i in range(len(data))]
        
        for tree in forest:
            probas = tree.predict_proba(data)
            for t, p in enumerate(probas):
                for key in p:
                    retProba[t].setdefault(key, 0)
                    retProba[t][key] += p[key] / len(forest)
                
        yList = list(set(y))
        yList.sort()
        ret = []
        
        for p in retProba:
            retAp = []
            for i in yList:
                if i not in p:
                    retAp.append(0.0)
                else:
                    retAp.append(p[i])
            ret.append(retAp)

        return np.array(ret)
        
    def score(self, data, y):
        forest = self.forest
        y_pred, error = self.validation(y, data, forest)
        return 1 - error
        
        

if __name__ == '__main__':
    filename = "/home/yejiming/desktop/Kaggle/OttoGroup/train.csv"
    df = pd.read_csv(filename, header = 0)
    df = df.drop(['id'], axis = 1)
    data = df.values
    np.random.shuffle(data)
    y = data[:, -1]
    y = labelEncoder(y)
    data = data[:, 0:-1]
    train = data[0:50000]
    ytrain = y[0:50000]
    test = data[50000:60000]
    ytest = y[50000:60000]

    learner = RandomForest()
    learner.train(train, ytrain, test, ytest)

    
