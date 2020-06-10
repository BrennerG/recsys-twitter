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

SCHEMA = StructType([
                StructField("tweet_id", StringType()),
                StructField("engaged_with_user_id", StringType()),
                StructField("engaging_user_id", StringType()),
                StructField("reply_timestamp", LongType()),
                StructField("retweet_timestamp", LongType()),
                StructField("retweet_with_comment_timestamp", LongType()),
                StructField("like_timestamp", LongType()),
                StructField("tweet_id_index", LongType()),
                StructField("engaged_with_user_id_index", LongType()),
                StructField("engaging_user_id_index", LongType())
])

class train_test_split:
    def __init__(self, spark: SparkSession, sc: SparkContext):
        self.spark = spark
        self.sc = sc
    
    def read_raw(self, inputFile: str, n_users: int = 0, n_tweets: int = 0):
        self.preproc = twitter_preproc(self.spark, self.sc, inputFile, MF=True)
        self.df = self.preproc.getDF()
        if n_users > 0:
            self.df = self.sample_users(n_users)
        if n_tweets > 0:
            self.df = self.sample_tweets(n_tweets)
        self.df = self.index_ids()
        
        #self.user_count = self.count_users()
        #self.tweet_count = self.count_tweets()

    def read_prepocessed(self, inputFile: str):
        self.df = self.spark.read.csv(inputFile, sep="\x01", header=True, schema=SCHEMA)

    def file_exists(self, file_name: str):
        fs = self.sc._jvm.org.apache.hadoop.fs.FileSystem.get(self.sc._jsc.hadoopConfiguration())
        return fs.exists(self.sc._jvm.org.apache.hadoop.fs.Path(file_name))

    def sample_users(self, n_users: int):
        user_ids = self.df\
            .select("engaging_user_id").distinct()\
            .sort("engaging_user_id").limit(n_users)
        return self.df\
            .join(user_ids, self.df["engaging_user_id"] == user_ids["engaging_user_id"])\
            .drop(user_ids["engaging_user_id"])
    
    def sample_tweets(self, n_tweets: int):
        tweet_ids = self.df\
            .select("tweet_id").distinct()\
            .sort("tweet_id").limit(n_tweets)
        return self.df\
            .join(tweet_ids, self.df["tweet_id"] == tweet_ids["tweet_id"])\
            .drop(tweet_ids["tweet_id"])
        
    def index_ids(self):
        indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in ["tweet_id", "engaging_user_id", "engaged_with_user_id"]]
        pipeline = Pipeline(stages=indexers)
        df = pipeline.fit(self.df).transform(self.df)
        return df.withColumn("tweet_id_index", df["tweet_id_index"].cast(LongType()))\
            .withColumn("engaged_with_user_id_index", df["engaged_with_user_id_index"].cast(LongType()))\
            .withColumn("engaging_user_id_index", df["engaging_user_id_index"].cast(LongType()))
    
    def count_users(self):
        return self.df.select("engaging_user_id_index")\
            .groupBy("engaging_user_id_index")\
            .agg(F.count("engaging_user_id_index").alias("n"))
    
    def count_tweets(self):
        return self.df.select("tweet_id_index")\
            .groupBy("tweet_id_index")\
            .agg(F.count("tweet_id_index").alias("n"))

    def get_all_occurring_twice(self):
        uc = self.user_count.filter(F.col("n") > 1)
        tc = self.tweet_count.filter(F.col("n") > 1)
        return self.df\
            .join(uc, self.df["engaging_user_id_index"] == uc["enganging_user_id_index"])\
            .join(tc, self.df["tweet_id_index"] == tc["tweet_id_index"])

    def write_to_csv(self, fileName: str):
        self.df.repartition(1).write.csv(fileName, sep="\x01", header=True)