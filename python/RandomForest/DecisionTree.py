import pandas as pd
from numpy import *
import numpy as np
import random
from math import sqrt
from sklearn import preprocessing
import time
from operator import itemgetter

from utils import *


class TreeNode(object):
    def __init__(self, feat, val, mc, probas, left = 0, right = 0):
        self.feature = feat
        self.value = val
        self.maxClass = mc
        self.left = left
        self.right = right
        self.probas = probas


class BTree(object):
    def __init__(self, root = 0):
        self.root = root
        
    def is_empty(self):
        if self.root is 0:
            return True

        else:
            return False
        
    def preOrder(self, treenode):
        if isinstance(treenode, str) or isinstance(treenode, int):
            return treenode

        print treenode.feature, treenode.value, treenode.maxClass
        self.preOrder(treenode.left)
        self.preOrder(treenode.right)
        
    def inOrder(self, treenode):
        if isinstance(treenode, str) or isinstance(treenode, int):
            return treenode

        self.inOrder(treenode.left)
        print treenode.feature, treenode.value, treenode.maxClass
        self.inOrder(treenode.right)
        
    def postOrder(self, treenode):
        if isinstance(treenode, str) or isinstance(treenode, int):
            return treenode

        self.postOrder(treenode.left)
        self.postOrder(treenode.right)
        print treenode.feature, treenode.value, treenode.maxClass

    def getNumLeafs(self, treenode):
        numLeafs = 0

        if treenode.left == 0 and treenode.right == 0:
            numLeafs += 1

        else:
            numLeafs += self.getNumLeafs(treenode.left)
            numLeafs += self.getNumLeafs(treenode.right)

        return numLeafs

    def getClass(self, x, treenode):
        if treenode.left == 0 and treenode.right == 0:
            return treenode.maxClass
        
        else:
            if x[treenode.feature] > treenode.value:
                return self.getClass(x, treenode.right)
            
            else:
                return self.getClass(x, treenode.left)

    def getProba(self, x, treenode):
        if treenode.left == 0 and treenode.right == 0:
            return treenode.probas
        
        else:
            if x[treenode.feature] > treenode.value:
                return self.getProba(x, treenode.right)
            
            else:
                return self.getProba(x, treenode.left)

        
class ClassificationTree(object):

    def __init__(self, tol=0.0001, leastSample=4,
                 maxDepth=inf, merge=None, maxFeatures=10):
        self.tol = tol
        self.leastSample = leastSample
        self.tree = BTree()
        self.maxDepth = maxDepth
        self.merge = merge
        self.maxFeatures = maxFeatures

    def chooseBestSplit(self, data, y, tol, leastSample):
        tol = self.tol
        leastSample = self.leastSample
        maxFeatures = self.maxFeatures
        
        if len(set(y)) == 1:
            return None, None, y[0], {y[0]: 1.0}
        
        m, n = shape(data)
        S = gini(y)
        bestS = inf
        bestIndex = 0
        bestValue = 0
        featureList = range(n)

        visitedFeatures = 0

        while (visitedFeatures < maxFeatures and len(featureList) >= 1):
            featIndex = random.sample(featureList, 1)[0]
            featureList.remove(featIndex)
            featSet = set(data[:, featIndex])
            lastnewS = inf
            
            if len(featSet) == 1:
                continue

            visitedFeatures += 1

            # Build the dictionary of y for each feature value
            # start = time.time()
            xd, yd = feat_map(data, y, featIndex, featSet)
            # print time.time() - start

            # whole = sorted(zip(data, y), key=itemgetter(featIndex), reverse=False)
            # data = whole[:, :-1]
            # y = whole[: -1]

            # Make the loop to find the best split value
            for split_val in featSet:
                # mat0, y0, mat1, y1 = bin_split(data, y, featIndex, split_val)
                length1 = length2 = 0
                for key in yd:
                    if int(key) <= split_val:
                        length1 += yd[key]
                    else:
                        length2 += yd[key]

                if (length1 < leastSample) or (length2 < leastSample):
                    continue
                
                r0 = float(length1) / len(y)
                r1 = float(length2) / len(y)

                d1 = xd[split_val][:]
                d2 = [0.0 for i in range(9)]
                for key in xd:
                    class_list = xd[key]
                    if int(key) < split_val:
                        for t, i in enumerate(class_list):
                            d1[t] += i
                    elif int(key) > split_val:
                        for t, i in enumerate(class_list):
                            d2[t] += i

                g1 = calc_gini(d1, length1)
                g2 = calc_gini(d2, length2)
                newS = r0 * g1 + r1 * g2

                if newS > lastnewS:
                    break
                
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = split_val
                    bestS = newS
                    
                lastnewS = newS

        if (S - bestS) < tol:
            return None, None, classify(y), probaDict(y)
        
        mat0, y0, mat1, y1 = bin_split(data, y, bestIndex, bestValue)

        if (shape(mat0)[0] < leastSample) or (shape(mat1)[0] < leastSample):
            return None, None, classify(y), probaDict(y)

        return bestIndex, bestValue, classify(y), probaDict(y)
        
    def createTree(self, data, y, tol, leastSample, depth = 0):
        tol = self.tol
        leastSample = self.leastSample
        retTree = TreeNode(-1, -1, -1, -1)
        
        if depth >= self.maxDepth:
            retTree.feature = None
            retTree.value = None
            retTree.maxClass = self.classify(y)
            retTree.probas = probaDict(y)
            return retTree
        
        feat, val, mc, probas = self.chooseBestSplit(data, y, tol, leastSample)
        if feat == None:
            retTree.feature = None
            retTree.value = None
            retTree.maxClass = mc
            retTree.probas = probas
            return retTree
        
        retTree.feature = feat
        retTree.value = val
        retTree.maxClass = mc
        retTree.probas = probas
        left, yleft, right, yright = bin_split(data, y, feat, val)
        retTree.left = self.createTree(left, yleft, tol, leastSample, depth + 1)
        retTree.right = self.createTree(right, yright, tol, leastSample, depth + 1)

        return retTree

    def train(self, data, y):
        tol = self.tol
        leastSample = self.leastSample
        length = len(y)
        train = data[:(length/10)*9]
        test = data[(length/10)*9:]
        ytrain = y[:(length/10)*9]
        ytest = y[(length/10)*9:]
        
        if self.merge == 'pep':
            root = self.createTree(data, y, tol, leastSample)
            self.tree = BTree(root)
            root = self.prune_pep(root, data, y)
            
        if self.merge == None:
            root = self.createTree(data, y, tol, leastSample)
            
        self.tree = BTree(root)

    def isLeaf(self, treenode):
        return (treenode.feature == None)

    def prune_pep(self, treenode, data, y):
        if self.isLeaf(treenode):
            return treenode
        
        numLeafs = self.tree.getNumLeafs(treenode)
        errorNoMerge = (1 - self.score(data, y)) * len(y) + 0.5 * numLeafs
        varNoMerge = sqrt(errorNoMerge * (1 - errorNoMerge / float(len(y))))
        y_pred = [treenode.maxClass] * len(y)
        errorMerge = len(y) - correctNum(y_pred, y) + 0.5
        
        if errorMerge < errorNoMerge + varNoMerge:
            treenode.feature = None
            treenode.value = None
            treenode.left = 0
            treenode.right = 0
            return treenode
        
        else:
            left, yleft, right, yright = bin_split(data, y,
                                            treenode.feature, treenode.value)

            if not self.isLeaf(treenode.left):
                treenode.left = self.prune_pep(treenode.left, left, yleft)

            if not self.isLeaf(treenode.right):
                treenode.right = self.prune_pep(treenode.right, right, yright)

            return treenode

    def predict(self, data):
        bt = self.tree
        ret = []
        
        for x in data:
            y_pred = bt.getClass(x, bt.root)
            ret.append(y_pred)
            
        ret = array(ret)
        return ret

    def predict_proba(self, data):
        bt = self.tree
        ret = []
        
        for x in data:
            probas = bt.getProba(x, bt.root)
            ret.append(probas)
            
        ret = array(ret)
        return ret

    def score(self, data, y):
        y_pred = self.predict(data)
        return float(correctNum(y, y_pred)) / len(y)
        

def calc_gini(d, total):
    to_square = list()
    for i in d:
        to_square.append(i/float(total))
    to_square = np.array(to_square)
    return 1 - (to_square * to_square.T).sum()

if __name__ == '__main__':
    filename = "dataset.csv"
    df = pd.read_csv(filename, header = 0)
    data = df.values
    y = data[:, -1]
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(y)
    data = data[:, 0:-1]
    train = data[0:50000]
    ytrain = y[0:50000]
    test = data[50000:60000]
    ytest = y[50000:60000]
    learner = ClassificationTree()
    learner.train(train, ytrain)
    s = learner.score(test, ytest)
    print s
