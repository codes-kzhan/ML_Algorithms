from itertools import *
import pandas as pd
from numpy import *
import random
from math import sqrt

class TreeNode(object):
    def __init__(self, feat, val, mc, left = 0, right = 0):
        self.feature = feat
        self.value = val
        self.maxClass = mc
        self.left = left
        self.right = right


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

        
class cart(object):

    def __init__(self, tol = 0.0001, leastSample = 1,
                 maxDepth = inf, merge = None):
        self.tol = tol
        self.leastSample = leastSample
        self.tree = BTree()
        self.maxDepth = maxDepth
        self.merge = merge
    
    def loadData(self, data):
        dataMat = []
        for t, row in enumerate(data):
            fltRow = map(float, row)
            dataMat.append(fltRow)
        dataMat = array(dataMat)
        return dataMat

    def gini(self, y):
        ret = 0
        y_dict = {}
        for i in y:
            y_dict.setdefault(i, 0)
            y_dict[i] += 1
        for key in y_dict:
            ret += (y_dict[key]/float(len(y))) ** 2
        ret = 1 - ret
        return ret

    def classify(self, y):
        y_dict = {}
        for i in y:
            y_dict.setdefault(i, 0)
            y_dict[i] += 1
        Max = 0
        for key in y_dict:
            if y_dict[key] > Max:
                Max = y_dict[key]
                MaxValue = key
        return MaxValue

    def chooseBestSplit(self, data, y, tol, leastSample):
        tol = self.tol
        leastSample = self.leastSample
        if len(set(y)) == 1:
            return None, None, y[0]
        m, n = shape(data)
        S = self.gini(y)
        bestS = inf; bestIndex = 0; bestValue = 0
        for featIndex in range(n):
            featSet = set(data[:,featIndex])
            lastnewS = inf
            if len(featSet) == 1:
                continue
            for splitVal in featSet:
                mat0, y0, mat1, y1 = self.binSplitDataSet(data, y, featIndex, splitVal)
                if (shape(mat0)[0] < leastSample) or (shape(mat1)[0] < leastSample):
                    continue
                r0 = float(len(y0)) / len(y)
                r1 = float(len(y1)) / len(y)
                newS = r0 * self.gini(y0) + r1 * self.gini(y1)
                if newS > lastnewS:
                    break
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
                lastnewS = newS
        if (S - bestS) < tol:
            return None, None, self.classify(y)
        mat0, y0, mat1, y1 = self.binSplitDataSet(data, y, bestIndex, bestValue)
        if (shape(mat0)[0] < leastSample) or (shape(mat1)[0] < leastSample):
            return None, None, self.classify(y)
        return bestIndex, bestValue, self.classify(y)
        
    def createTree(self, data, y, tol, leastSample, depth = 0):
        tol = self.tol
        leastSample = self.leastSample
        retTree = TreeNode(-1, -1, -1)
        if depth >= self.maxDepth:
            retTree.feature = None
            retTree.value = None
            retTree.maxClass = self.classify(y)
            return retTree
        feat, val, mc = self.chooseBestSplit(data, y, tol, leastSample)
        if feat == None:
            retTree.feature = None
            retTree.value = None
            retTree.maxClass = mc
            return retTree
        retTree.feature = feat
        retTree.value = val
        retTree.maxClass = mc
        left, yleft, right, yright = self.binSplitDataSet(data, y, feat, val)
        retTree.left = self.createTree(left, yleft, tol, leastSample, depth + 1)
        retTree.right = self.createTree(right, yright, tol, leastSample, depth + 1)
        return retTree

    def binSplitDataSet(self, data, y, feature, value):
        determine = data[:,feature] <= value
        mat0 = data[determine]  
        mat1 = data[determine == False]
        y0 = y[determine]
        y1 = y[determine == False]
        return mat0, y0, mat1, y1

    def train(self, data, y):
        tol = self.tol
        leastSample = self.leastSample
        length = len(y)
        data = self.loadData(data)
        train = data[:(length/10)*9]
        test = data[(length/10)*9:]
        ytrain = y[:(length/10)*9]
        ytest = y[(length/10)*9:]
        if self.merge == 'pep':
            root = self.createTree(data, y, tol, leastSample)
            self.tree = BTree(root)
            root = self.prune_pep(root, data, y)
        if self.merge == 'rep':
            root = self.createTree(train, ytrain, tol, leastSample)
            root = self.prune(root, test, ytest)
        if self.merge == None:
            root = self.createTree(data, y, tol, leastSample)
        self.tree = BTree(root)

    def correctNum(self, y, y_pred):
        ret = 0
        for t, i in enumerate(y):
            if i == y_pred[t]:
                ret += 1
        return ret

    def isLeaf(self, treenode):
        return (treenode.feature == None)

    def prune(self, treenode, test, ytest):
        if not self.isLeaf(treenode.left) or not self.isLeaf(treenode.right):
            left, yleft, right, yright = self.binSplitDataSet(test, ytest,
                                            treenode.feature, treenode.value)
        if not self.isLeaf(treenode.left):
            treenode.left = self.prune(treenode.left, left, yleft)
        if not self.isLeaf(treenode.right):
            treenode.right = self.prune(treenode.right, right, yright)
        if self.isLeaf(treenode.left) and self.isLeaf(treenode.right):
            left, yleft, right, yright = self.binSplitDataSet(test, ytest,
                                            treenode.feature, treenode.value)
            yleft_pred = [treenode.left.maxClass] * len(yleft)
            yright_pred = [treenode.right.maxClass] * len(yright)
            correctNoMerge = self.correctNum(yleft_pred, yleft) + \
                           self.correctNum(yright_pred, yright)
            y_pred = [treenode.maxClass] * len(ytest)
            correctMerge = self.correctNum(y_pred, ytest)
            if correctMerge > correctNoMerge:
                print "Merging"
                treenode.feature = None
                treenode.value = None
                treenode.left = 0
                treenode.right = 0
                return treenode
            else:
                return treenode
        return treenode

    def prune_pep(self, treenode, data, y):
        if self.isLeaf(treenode):
            return treenode
        numLeafs = self.tree.getNumLeafs(treenode)
        errorNoMerge = (1 - self.score(data, y)) * len(y) + 0.5 * numLeafs
        varNoMerge = sqrt(errorNoMerge * (1 - errorNoMerge / float(len(y))))
        y_pred = [treenode.maxClass] * len(y)
        errorMerge = len(y) - self.correctNum(y_pred, y) + 0.5
        if errorMerge < errorNoMerge + sqrt(varNoMerge):
            print "Merging"
            treenode.feature = None
            treenode.value = None
            treenode.left = 0
            treenode.right = 0
            return treenode
        else:
            left, yleft, right, yright = self.binSplitDataSet(data, y,
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
        return ret

    def score(self, data, y):
        y_pred = self.predict(data)
        return float(self.correctNum(y, y_pred)) / len(y)
        
        
if __name__ == '__main__':
    filename = "/home/yejiming/桌面/Kaggle/OttoGroup/train.csv"
    df = pd.read_csv(filename, header = 0)
    df = df.drop(['id'], axis = 1)
    data = df.values
    random.shuffle(data)
    y = data[:, -1]
    data = data[:, 0:-1]
    train = data[0:50000]
    ytrain = y[0:50000]
    test = data[50000:60000]
    ytest = y[50000:60000]
    learner = cart()
    learner.train(train, ytrain)
    
