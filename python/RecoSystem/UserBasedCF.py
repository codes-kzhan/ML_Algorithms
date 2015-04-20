import pandas as pd
import numpy as np
import csv
from scipy.stats.stats import pearsonr
from operator import itemgetter
import pymongo
from pymongo import MongoClient

from LSH import *


class Database:
    "A class representing a database of similaries and common supports"
    
    def __init__(self, df):
        "the constructor, takes a reviews dataframe like smalldf as its argument"
        database={}
        self.df=df
        self.uniquebizids={v:k for (k,v) in enumerate(df.business_id.unique())}
        keys=self.uniquebizids.keys()
        l_keys=len(keys)
        self.database_sim=np.zeros([l_keys,l_keys])
        self.database_sup=np.zeros([l_keys, l_keys], dtype=np.int)
        
    def populate_by_calculating(self, similarity_func):
        """
        a populator for every pair of businesses in df. takes similarity_func like
        pearson_sim as argument
        """
        items=self.uniquebizids.items()
        for b1, i1 in items:
            for b2, i2 in items:
                if i1 < i2:
                    sim, nsup=calculate_similarity(b1, b2, self.df, similarity_func)
                    self.database_sim[i1][i2]=sim
                    self.database_sim[i2][i1]=sim
                    self.database_sup[i1][i2]=nsup
                    self.database_sup[i2][i1]=nsup
                elif i1==i2:
                    nsup=self.df[self.df.business_id==b1].user_id.count()
                    self.database_sim[i1][i1]=1.
                    self.database_sup[i1][i1]=nsup
                    

    def get(self, b1, b2):
        "returns a tuple of similarity,common_support given two business ids"
        sim=self.database_sim[self.uniquebizids[b1]][self.uniquebizids[b2]]
        nsup=self.database_sup[self.uniquebizids[b1]][self.uniquebizids[b2]]
        return (sim, nsup)


def get_user_reviews(user_id, df):
    """
    given a user id and a set of restaurants, return the sub-dataframe of their
    reviews.
    """
    set_of_rests = df.business_id.unique()
    mask = (df.business_id.isin(set_of_rests)) & (df.user_id == user_id)
    reviews = df[mask]
    reviews = reviews[reviews.business_id.duplicated() == False]
    return reviews

def create_user_biz(df):
    uniqueuserid = df.user_id.unique()
    uniquebizid = df.business_id.unique()
    filename = open("user_biz.csv", "wb")
    open_file_object = csv.writer(filename)
    rows = []
    header = ["user_id"]
    header.extend([i for i in uniquebizid])
    open_file_object.writerow(header)
    for t, userid in enumerate(uniqueuserid):
        df = get_user_reviews(userid, fulldf)
        bizids = df.business_id.values
        stars = fulldf.stars.values
        row = [userid]
        row.extend([0 for i in range(len(uniquebizid))])
        for bizid, star in zip(bizids, stars):
            index = header.index(bizid)
            row[index] = star
        rows.append(row)
        if t % 100 == 0 and t != 0:
            print t
            open_file_object.writerows(rows)
            rows = []
    filename.close()

def import_user_biz(df):
    uniqueuserid = df.user_id.unique()
    uniquebizid = df.business_id.unique()
    header = ["user_id"]
    header.extend([i for i in uniquebizid])
    for t, userid in enumerate(uniqueuserid):
        if t == 0:
            continue
        df = get_user_reviews(userid, fulldf)
        bizids = df.business_id.values
        stars = fulldf.stars.values
        row = [0 for i in range(len(uniquebizid))]
        for bizid, star in zip(bizids, stars):
            index = header.index(bizid)
            row[index - 1] = star
        user_biz.insert({'user_id':userid, 'vector':row})
        if t % 100 == 0:
            print t
    user_biz.ensure_index('user_id', unique=True)

if __name__ == '__main__':
    fulldf=pd.read_csv("bigdf.csv")
    
    connection = MongoClient('localhost', 27017)
    db = connection.RecoSystem
    user_biz = db.user_biz
##    import_user_biz(fulldf)

    lsh = CosineNN(4503)
    lsh.populate(user_biz.find())
