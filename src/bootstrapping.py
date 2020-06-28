import pandas as pd
from sklearn.utils import resample
from typing import List, Dict

from pyspark import SparkConf, SparkContext
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from IPython.display import display

class bootstrapping:

    def __init__(self, spark: SparkSession, sc: SparkContext, file_name: str):
        self.spark = spark
        self.sc = sc

        SCHEMA = StructType([
                StructField("text_tokens", StringType()),
                StructField("hashtags", StringType()),
                StructField("tweet_id", StringType()),
                StructField("present_media", StringType()),
                StructField("present_links", StringType()),
                StructField("present_domains", StringType()),
                StructField("tweet_type", StringType()),
                StructField("language", StringType()),
                StructField("tweet_timestamp", LongType()),
                StructField("engaged_with_user_id", StringType()),
                StructField("engaged_with_user_follower_count", LongType()),
                StructField("engaged_with_user_following_count", LongType()),
                StructField("engaged_with_user_is_verified", BooleanType()),
                StructField("engaged_with_user_account_creation", LongType()),
                StructField("engaging_user_id", StringType()),
                StructField("engaging_user_follower_count", LongType()),
                StructField("engaging_user_following_count", LongType()),
                StructField("engaging_user_is_verified", BooleanType()),
                StructField("engaging_user_account_creation", LongType()),
                StructField("engaged_follows_engaging", BooleanType()),
                StructField("reply_timestamp", LongType()),
                StructField("retweet_timestamp", LongType()),
                StructField("retweet_with_comment_timestamp", LongType()),
                StructField("like_timestamp", LongType())       
            ])
        
        self.df = self.spark.read.csv(path=file_name, sep="\x01", header=False, schema=SCHEMA)
        
        zipped = self.df.rdd.zipWithIndex().toDF()
        self.df = zipped.select([F.col("_2").alias("id")] + [F.col("_1")[column_name].alias(column_name) for column_name in self.df.columns])
        self.df_count = self.df.count()

    def sample(self, n_samples: int, seed: int):
        sample_list = resample(range(self.df_count), n_samples=n_samples, random_state=seed)
        R = Row("id")
        samples = self.spark.createDataFrame([R(i) for i in sample_list])
        return samples.join(self.df, "id")

    def write_to_csv(self, df, file_name: str):
        df.drop("id").coalesce(1).write.csv(file_name, sep="\x01", header=False)