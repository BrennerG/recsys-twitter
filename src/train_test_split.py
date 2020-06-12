import twitter_preproc
import importlib
importlib.reload(twitter_preproc)
from twitter_preproc import *

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.sql.types import *

"""
Workflow: Read -> 
"""
class train_test_split:

    def ENGAGEMENTS(self):
        return ["like", "reply", "retweet", "retweet_with_comment"]

    def __init__(self, spark: SparkSession, sc: SparkContext):
        self.spark = spark
        self.sc = sc
    
    def read_raw(self, inputFile: str, n_users: int = 0, n_tweets: int = 0, full=True):
        self.preproc = twitter_preproc(self.spark, self.sc, inputFile, MF=True)
        self.df = self.preproc.getDF()
        if n_users > 0:
            self.df = self.sample_users(self.df, n_users)
        if n_tweets > 0:
            self.df = self.sample_tweets(self.df, n_tweets)
        
        self.df = self.index_ids(self.df)
        if full:
            self.full_df = self.get_full(self.df)
            self.full_df = self.timestamps_to_boolean(self.full_df)
        else:
            self.df = self.timestamps_to_boolean(self.df)

    def read_preprocessed(self, inputFile: str):
        self.df = self.spark.read.csv(inputFile, sep="\x01", header=True)
        cast_columns = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp", "tweet_id_index", 
            "engaged_with_user_id_index", "engaging_user_id_index"]
        for cast_column in cast_columns:
            self.df = self.df.withColumn(cast_column, self.df[cast_column].cast(LongType()))
        
        self.full_df = self.get_full(self.df)
        self.full_df = self.timestamps_to_boolean(self.full_df)

    def sample_users(self, df, n_users: int):
        user_ids = df\
            .select("engaging_user_id").distinct()\
            .sort("engaging_user_id").limit(n_users)
        return df\
            .join(user_ids, df["engaging_user_id"] == user_ids["engaging_user_id"])\
            .drop(user_ids["engaging_user_id"])
    
    def sample_tweets(self, df, n_tweets: int):
        tweet_ids = df\
            .select("tweet_id").distinct()\
            .sort("tweet_id").limit(n_tweets)
        return df\
            .join(tweet_ids, df["tweet_id"] == tweet_ids["tweet_id"])\
            .drop(tweet_ids["tweet_id"])
        
    def index_ids(self, df):
        indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in ["tweet_id", "engaging_user_id", "engaged_with_user_id"]]
        pipeline = Pipeline(stages=indexers)
        df = pipeline.fit(df).transform(df)
        return df.withColumn("tweet_id_index", df["tweet_id_index"].cast(LongType()))\
            .withColumn("engaged_with_user_id_index", df["engaged_with_user_id_index"].cast(LongType()))\
            .withColumn("engaging_user_id_index", df["engaging_user_id_index"].cast(LongType()))
    
    def timestamps_to_boolean(self, df):
        for engagement in self.ENGAGEMENTS():
            df = df.withColumn(engagement, when(df[engagement + "_timestamp"].isNotNull(), 1))
        return df.fillna(0, subset=self.ENGAGEMENTS())

    def get_full(self, df):
        user_ids = df.select("engaging_user_id_index").distinct()
        tweet_ids = df.select("tweet_id_index").distinct()
        cross_joined = user_ids.join(tweet_ids)
        return cross_joined.join(df, ["engaging_user_id_index", "tweet_id_index"], how="left")

    def get_all_train_test(self, df, train_split=0.8, neg_ratio=0.5):
        neg = self.get_negatives(df)
        neg_train, neg_test = neg.randomSplit([train_split, 1 - train_split])
        
        self.train = {}
        self.test = {}
        for engagement in self.ENGAGEMENTS():
            pos = self.get_positives(df, engagement)
            pos_train, pos_test = pos.randomSplit([train_split, 1 - train_split])
            
            self.train[engagement] = pos_train.union(neg_train)
            self.test[engagement] = pos_test.union(neg_test.limit(int(pos_test.count() * neg_ratio)))
            

    def get_positives(self, df, engagement: str):
        if engagement not in self.ENGAGEMENTS():
            return df
        return df.filter(F.col(engagement + "_timestamp").isNotNull())

    def get_negatives(self, df):
        return df\
            .filter(F.col("like_timestamp").isNull())\
            .filter(F.col("reply_timestamp").isNull())\
            .filter(F.col("retweet_timestamp").isNull())\
            .filter(F.col("retweet_with_comment_timestamp").isNull())

    def write_to_csv(self, df, fileName: str):
        df.repartition(1).write.csv(fileName, sep="\x01", header=True)