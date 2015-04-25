import pandas as pd
import numpy as np
import csv
from scipy.stats.stats import pearsonr
from operator import itemgetter
import pymongo
from pymongo import MongoClient

from LSH import *

connection = MongoClient('localhost', 27017)
db = connection.RecoSystem

def recompute_frame(ldf):
    """
    takes a dataframe ldf, makes a copy of it, and returns the copy
    with all averages and review counts recomputed
    this is used when a frame is subsetted.
    """
    ldfu=ldf.groupby('user_id')
    ldfb=ldf.groupby('business_id')
    user_avg=ldfu.stars.mean()
    user_review_count=ldfu.review_id.count()
    business_avg=ldfb.stars.mean()
    business_review_count=ldfb.review_id.count()
    nldf=ldf.copy()
    nldf.set_index(['business_id'], inplace=True)
    nldf['business_avg']=business_avg
    nldf['business_review_count']=business_review_count
    nldf.reset_index(inplace=True)
    nldf.set_index(['user_id'], inplace=True)
    nldf['user_avg']=user_avg
    nldf['user_review_count']=user_review_count
    nldf.reset_index(inplace=True)
    return nldf

class Database:
    "A class representing a database of each user's neighbourhoods and ratings"
    
    def __init__(self, df):
        self.neighbour = db.cf_neighbour
        self.lsh = LSH(4503)
        self.users = df.user_id.unique()
        self.df = df
        
    def populate_by_calculating(self, k=5):
        """
        a populator for generating each user's top k nearest neighbourhoods.
        """
        users = self.users
        for t, user_id in enumerate(users):
            topK = self.lsh.topK(user_id)
            self.neighbour.insert({'_id':user_id, 'neighbours':topK})
            if t % 100 == 0:
                print t
        

##    def rating(restaurant_id, user_id, k=7, reg=3.):


def get_user_reviews(user_id, df):
    """
    given a user id, return the sub-dataframe of their reviews.
    """
    set_of_rests = df.business_id.unique()
    mask = (df.business_id.isin(set_of_rests)) & (df.user_id == user_id)
    reviews = df[mask]
    reviews = reviews[reviews.business_id.duplicated() == False]
    return reviews

def get_restaurant_reviews(restaurant_id, df):
    """
    given a resturant id, return the sub-dataframe of their reviews.
    """
    set_of_users = df.user_id.unique()
    mask = (df.user_id.isin(set_of_users)) & (df.business_id==restaurant_id)
    reviews = df[mask]
    reviews = reviews[reviews.user_id.duplicated()==False]
    return reviews

def import_user_biz(df):
    uniqueuserid = df.user_id.unique()
    uniquebizid = df.business_id.unique()
    header = ["user_id"]
    header.extend([i for i in uniquebizid])
    for t, userid in enumerate(uniqueuserid):
        if t == 0:
            continue
        df2 = get_user_reviews(userid, df)
        username = df2.user_name.values[0]
        bizids = df2.business_id.values
        stars = df2.stars.values
        row = [0 for i in range(len(uniquebizid))]
        for bizid, star in zip(bizids, stars):
            index = header.index(bizid)
            row[index - 1] = star
        user_biz.insert({'_id':userid, 'user_name':username, 'vector':row})
        if t % 100 == 0:
            print t

def import_biz_user(df):
    uniqueuserid = df.user_id.unique()
    uniquebizid = df.business_id.unique()
    header = ["biz_id"]
    header.extend([i for i in uniqueuserid])
    for t, bizid in enumerate(uniquebizid):
        df2 = get_restaurant_reviews(bizid, df)
        bizname = df2.biz_name.values[0]
        userids = df2.user_id.values
        stars = df2.stars.values
        average = df2.user_avg.values
        row = [0 for i in range(len(uniqueuserid))]
        for t, userid in enumerate(userids):
            index = header.index(userid)
            row[index - 1] = stars[t] - average[t] 
        biz_user.insert({'_id':bizid, 'biz_name':bizname, 'vector':row})
        if t % 100 == 0:
            print t

if __name__ == '__main__':
    fulldf=pd.read_csv("bigdf.csv")
    smallidf=fulldf[(fulldf.user_review_count > 60) & (fulldf.business_review_count > 150)]
    smalldf=recompute_frame(smallidf)
    
    biz_user = db.biz_user
    import_biz_user(smalldf)

    lsh = LSH(240)
    lsh.populate(biz_user.find())
