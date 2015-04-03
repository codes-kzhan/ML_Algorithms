from itertools import *
import pandas as pd
from numpy import *

filename = "/home/yejiming/桌面/Kaggle/OttoGroup/train.csv"
df = pd.read_csv(filename, header = 0)
y = df.target.values
df = df.drop(['target', 'id'], axis = 1)
data = df.values

class TreeNode(object):
    def __init__(self, feat, val, left = 0, right = 0):
        self.feature = feat
        self.value = val
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
        print treenode.feature, treenode.value
        self.preOrder(treenode.left)
        self.preOrder(treenode.right)
        
    def inOrder(self, treenode):
        if isinstance(treenode, str) or isinstance(treenode, int):
            return treenode
        self.inOrder(treenode.left)
        print treenode.feature, treenode.value
        self.inOrder(treenode.right)
        
    def postOrder(self, treenode):
        if isinstance(treenode, str) or isinstance(treenode, int):
            return treenode
        self.postOrder(treenode.left)
        self.postOrder(treenode.right)
        print treenode.feature, treenode.value

        
class cart(object):

    def __init__(self, tol = 0.01, leastSample = 10):
        self.tol = tol
        self.leastSample = leastSample
        self.tree = BTree()
    
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
            return None, y[0]
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
            return None, self.classify(y)
        mat0, y0, mat1, y1 = self.binSplitDataSet(data, y, bestIndex, bestValue)
        if (shape(mat0)[0] < leastSample) or (shape(mat1)[0] < leastSample):
            return None, self.classify(y)
        return bestIndex, bestValue
        
    def createTree(self, data, y, tol, leastSample, depth = 0):
        tol = self.tol
        leastSample = self.leastSample
        retTree = TreeNode(-1, -1)
        if depth >= 10:
            retTree.value = self.classify(y)
            return retTree
        feat, val = self.chooseBestSplit(data, y, tol, leastSample)
        if feat == None:
            retTree.value = val
            return retTree
        retTree.feature = feat
        retTree.value = val
        left, yleft, right, yright = self.binSplitDataSet(data, y, feat, val)
        retTree.left = self.createTree(left, yleft, tol, leastSample, depth + 1)
        retTree.right = self.createTree(right, yright, tol, leastSample, depth + 1)
        return retTree

    def featuresplit(features):   
        count = len(features)   
        featureind = range(count)   
        featureind.pop(0) #get value 1~(count-1)  
        combiList = []   
        for i in featureind:   
            com = list(combinations(features, len(features[0:i])))   
            combiList.extend(com)   
        combiLen = len(combiList)
        featuresplitGroup = zip(combiList[0:combiLen/2], combiList[combiLen-1:combiLen/2-1:-1])   
        return featuresplitGroup

    def splitDataSet(dataSet, axis, valueTuple):   
        '''return dataset satisfy condition dataSet[i][axis] == valueTuple, 
        and remove dataSet[i][axis] if len(valueTuple)==1'''  
        retDataSet = []   
        length = len(valueTuple)   
        if length ==1:   
          for featVec in dataSet:   
            if featVec[axis] == valueTuple[0]:   
                reducedFeatVec = featVec[:axis]     #chop out axis used for splitting  
                reducedFeatVec.extend(featVec[axis+1:])   
                retDataSet.append(reducedFeatVec)   
        else:   
          for featVec in dataSet:   
            if featVec[axis] in valueTuple:   
                retDataSet.append(featVec)   
        return retDataSet

    def binSplitDataSet(self, data, y, feature, value):
        determine = data[:,feature] > value
        mat0 = data[determine]  
        mat1 = data[determine == False]
        y0 = y[determine]
        y1 = y[determine == False]
        return mat0, y0, mat1, y1

    def fit(self, data, y):
        tol = self.tol
        leastSample = self.leastSample

        data = self.loadData(data)
        root = self.createTree(data, y, tol, leastSample)
        self.tree = BTree(root)
        
        

if __name__ == '__main__':   
    t = cart()
    t.fit(data, y)
    
