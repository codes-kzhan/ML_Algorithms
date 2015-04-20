import pandas as pd
import numpy as np
import csv
from scipy.stats.stats import pearsonr
from operator import itemgetter

from LSH import *


def pearson_sim(vec1, vec2):
    return pearsonr(vec1, vec2)[0]

def knearest(Id, k=5, reg=3.):
    vec1 = np.array(table.col_vecs[Id])
    similars = []
    candidate = table.query(Id)
    for other_id in candidate:
        if other_id != Id:
            vec2 = np.array(table.col_vecs[other_id])
            sim = pearson_sim(vec1, vec2)
            similars.append((other_id, sim))
    similars = sorted(similars, key=itemgetter(1), reverse=True)
    return similars[0:k]

##def rating(Id):
##    similars = knearest(Id)
##    ret = 0
##    for item in similars:
##        yi = y[item[0]]
##        ret
    
if __name__ == '__main__':
    train_df = pd.read_csv(r'train.csv', header = 0)
    y = train_df['target'].values
    train_df = train_df.drop(['target', 'id'], axis=1)
    train_data = train_df.values
    table = CosineNN(93)
    table.populate(train_data)
