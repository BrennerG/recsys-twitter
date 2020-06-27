import pyspark.sql.functions as Fun
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import twitter_preproc
import importlib
importlib.reload(twitter_preproc)
from twitter_preproc import *
from operator import attrgetter
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from twitter_preproc import twitter_preproc
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
import torch
import sys, traceback

class NNPreprocessor:

    def __init__(self):
        pass
    
    def get_id_indices(self, df, id_column):
        id_indices = df.select(id_column).orderBy(id_column).rdd.zipWithIndex().toDF()
        id_indices = id_indices.withColumn(id_column, Fun.col("_1")[id_column])\
            .select(Fun.col(id_column), Fun.col("_2").alias(id_column + "_index"))
        return id_indices

    # map function to convert from spark vectors to sparse numpy csr matrix
    def as_matrix(self,vec):
        data, indices = vec.values, vec.indices
        shape = 1, vec.size
        return csr_matrix((data, indices, np.array([0, vec.values.size])), shape)

    # calls as_matrix
    def get_pytorch_sparse(self, attr, traindata_ohe):
        features = traindata_ohe.rdd.map(attrgetter(attr))
        mats = features.map(lambda vec: csr_matrix((vec.values, vec.indices, np.array([0, vec.values.size])), (1,vec.size)))
        mat = mats.reduce(lambda x, y: vstack([x, y]))
        return mat

    # convert to pytorch format
    def transform_to_sparse_tensor(self,data):
        coo = data.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def nn_preprocess(self, data, engagement='like'):
        traindata = data

        # INDEXING & ONE-HOT ENCODING
        # get indexed columns
        tweet_id_idx = self.get_id_indices(df=traindata, id_column="tweet_id")
        user_id_idx = self.get_id_indices(df=traindata, id_column="engaging_user_id")
        # rejoin the columns
        indexed_data = traindata.join(tweet_id_idx, ['tweet_id']).join(user_id_idx, ['engaging_user_id'])

        # one-hot-encode
        pipeline = Pipeline(stages=[
            OneHotEncoder(inputCol="tweet_id_index",  outputCol="tweet_id_ohe"),
            OneHotEncoder(inputCol="engaging_user_id_index",  outputCol="user_id_ohe")
        ])
        model = pipeline.fit(indexed_data.select(['tweet_id_index', 'engaging_user_id_index', engagement]))
        ohe = model.transform(indexed_data)

        # select and parse to pandas dataframe
        df = pd.DataFrame(ohe.select(['tweet_id_ohe', 'user_id_ohe', engagement]).collect(), columns=['tweet_id_ohe', 'user_id_ohe', engagement])

        # create tweets and users vector in correct format
        tweet_sparse = self.get_pytorch_sparse("tweet_id_ohe", ohe)
        user_sparse = self.get_pytorch_sparse("user_id_ohe", ohe)
        tweets = self.transform_to_sparse_tensor(tweet_sparse).to_dense()
        users = self.transform_to_sparse_tensor(user_sparse).to_dense()

        # create target variables in correct format
        y = torch.FloatTensor(ohe.select(engagement).collect()) 
        target = y  
        target = target.view(1, -1).t()

        return tweets, users, target