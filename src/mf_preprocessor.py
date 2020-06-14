import twitter_preproc
import importlib
importlib.reload(twitter_preproc)
from twitter_preproc import *

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *

class mf_preprocessor:

    def ENGAGEMENTS(self):
        return ["like", "reply", "retweet", "retweet_with_comment"]

    def __init__(self, spark: SparkSession, sc: SparkContext):
        self.spark = spark
        self.sc = sc
    
    def read_raw(self, inputFile: str, n_users: int = 0, n_tweets: int = 0):
        preproc = twitter_preproc(self.spark, self.sc, inputFile, MF=True)
        df = preproc.getDF()
        df = df.drop("engaged_with_user_id")
        if n_users > 0:
            df = self.sample_users(df, n_users)
        if n_tweets > 0:
            df = self.sample_tweets(df, n_tweets)
        
        return self.timestamps_to_boolean(df)

    def read_preprocessed(self, inputFile: str):
        df = self.spark.read.csv(inputFile, sep="\x01", header=True)
        byte_columns = self.ENGAGEMENTS()
        for byte_column in byte_columns:
            df = df.withColumn(byte_column, df[byte_column].cast(ByteType()))
        long_columns = ["tweet_id_index", "engaging_user_id_index"]
        for long_column in long_columns:
            df = df.withColumn(long_column, df[long_column].cast(IntegerType()))
        
        return df

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

    def timestamps_to_boolean(self, df):
        for engagement in self.ENGAGEMENTS():
            df = df.withColumn(engagement, when(df[engagement + "_timestamp"].isNotNull(), 1).cast(ByteType()))\
                .drop(engagement + "_timestamp")
        return df.fillna(0, subset=self.ENGAGEMENTS())

    def get_id_indices(self, df, id_column):
        id_indices = df.select(id_column).orderBy(id_column).rdd.zipWithIndex().toDF()
        id_indices = id_indices.withColumn(id_column, F.col("_1")[id_column])\
                .select(F.col(id_column), F.col("_2").alias(id_column + "_index"))
        return id_indices

    def write_to_csv(self, df, fileName: str, single_file: bool = True):
        if single_file:
            df.repartition(1).write.csv(fileName, sep="\x01", header=True)
        else:
            df.write.csv(fileName, sep="\x01", header=True)