import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from IPython.display import display

import mf_preprocessor
import importlib
importlib.reload(mf_preprocessor)
from mf_preprocessor import *

class matrix_factorization:

    def ENGAGEMENTS(self):
        return self.preproc.ENGAGEMENTS()

    def __init__(self, spark: SparkSession, sc: SparkContext):
        self.spark = spark
        self.sc = sc
        self.preproc = mf_preprocessor(self.spark, self.sc)

    def read_train(self, train_file: str, is_preprocessed=True, n_users: int = 0, n_tweets: int = 0):
        self.train = self.read_file(train_file, is_preprocessed, n_users, n_tweets, delete_ids="train")

    def read_test(self, test_file: str, is_preprocessed=True, n_users: int = 0, n_tweets: int = 0):
        self.test = self.read_file(test_file, is_preprocessed, n_users, n_tweets)

    def read_val(self, val_file: str, is_preprocessed=True, n_users: int = 0, n_tweets: int = 0):
        self.val = self.read_file(val_file, is_preprocessed, n_users, n_tweets, delete_ids="val")

    def read_file(self, file_path: str, is_preprocessed: bool = True, n_users: int = 0, n_tweets: int = 0, delete_ids=None):
        if is_preprocessed:
            return self.preproc.read_preprocessed(file_path)
        return self.preproc.read_raw(file_path, n_users, n_tweets, delete_ids)

    def index_ids(self, df_train, df_test, index_files: dict = None):
        self.id_indices = {}
        id_columns = ["tweet_id", "engaging_user_id"]
        for id_column in id_columns:
            if index_files is None or id_column not in index_files:
                df_id = df_train.unionByName(df_test).select(id_column).distinct()
                self.id_indices[id_column] = self.preproc.get_id_indices(df_id, id_column)
            else:
                self.id_indices[id_column] = self.spark.read.csv(path=index_files[id_column], sep="\x01", header=True)
                self.id_indices[id_column] = self.id_indices[id_column].withColumn(id_column + "_index", F.col(id_column + "_index").cast(LongType()))
            
            df_train = df_train.join(self.id_indices[id_column], [id_column]).drop(id_column)
            df_test = df_test.join(self.id_indices[id_column], [id_column]).drop(id_column)
        return df_train, df_test

    def build_full_matrix(self, df_train, df_test, cross_join=False):
        if cross_join:
            user_ids = df_train.unionByName(df_test).select("engaging_user_id_index").distinct()
            tweet_ids = df_train.unionByName(df_test).select("tweet_id_index").distinct()
            cross_joined = user_ids.join(tweet_ids)
            self.full_train = cross_joined.join(df_train, ["engaging_user_id_index", "tweet_id_index"], how="left")
        else:
            missing_tweet_indices = df_test.select("tweet_id_index").distinct()\
                .subtract(df_train.select("tweet_id_index").distinct())
            missing_user_indices = df_test.select("engaging_user_id_index").distinct()\
                .subtract(df_train.select("engaging_user_id_index").distinct())
            
            index_pairs = self.concat_indices(missing_tweet_indices, missing_user_indices)
            for engagement in self.ENGAGEMENTS():
                index_pairs = index_pairs.withColumn(engagement, F.lit(0).cast(ByteType()))
            
            self.full_train = df_train.unionByName(index_pairs)
            
    def concat_indices(self, tweet_indices, user_indices):
        index_pairs = None
        n_tweet_indices = tweet_indices.count()
        n_user_indices = user_indices.count()
        if n_tweet_indices >= n_user_indices:
            user_join_index = user_indices.rdd.zipWithIndex().toDF()
            user_join_index = user_join_index.withColumn("engaging_user_id_index", F.col("_1")["engaging_user_id_index"])\
                                .select(F.col("engaging_user_id_index"), F.col("_2").alias("join_index"))
            while n_tweet_indices > 0:
                zipping_tweets = tweet_indices.limit(n_user_indices)
                tweet_indices = tweet_indices.subtract(zipping_tweets)
                n_tweet_indices -= n_user_indices

                tweets_join_index = zipping_tweets.rdd.zipWithIndex().toDF()
                tweets_join_index = tweets_join_index.withColumn("tweet_id_index", F.col("_1")["tweet_id_index"])\
                                .select(F.col("tweet_id_index"), F.col("_2").alias("join_index"))
                joined = user_join_index.join(tweets_join_index, ["join_index"]).drop("join_index")
                if index_pairs is None:
                    index_pairs = joined
                else:
                    index_pairs = index_pairs.unionByName(joined)
        else:
            raise NotImplementedError("Do implement")
            
        return index_pairs
                    

    def train_test_split(self, df, train_split=0.8, neg_ratio=1):
        neg = self.get_negatives(df)
        neg_train, neg_test = neg.randomSplit([train_split, 1 - train_split])
        
        self.train_sets = {}
        self.test_sets = {}
        for engagement in self.ENGAGEMENTS():
            pos = self.get_positives(df, engagement)
            pos_train, pos_test = pos.randomSplit([train_split, 1 - train_split])
            
            self.train_sets[engagement] = pos_train.unionByName(neg_train)
            self.test_sets[engagement] = pos_test.unionByName(neg_test.limit(int(pos_test.count() * neg_ratio)))

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

    def train_evaluate(self, engagement):
        (training, test) = self.train_sets[engagement], self.test_sets[engagement]
        model, predictions = self.fit_transform(training, test, engagement)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol=engagement, predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(rmse))
        return predictions

    def train_predict(self, df_train, df_test, engagement, rank=10):
        model, predictions = self.fit_transform(df_train, df_test, engagement, rank)
        return model, predictions

    def fit_transform(self, training, test, engagement, rank=10):
        # Build the recommendation model using ALS on the training data
        # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        als = ALS(rank=rank, maxIter=5, regParam=0.01, userCol="engaging_user_id_index", itemCol="tweet_id_index", ratingCol=engagement,
                coldStartStrategy="drop", implicitPrefs=True)
        model = als.fit(training)

        # Evaluate the model by computing the RMSE on the test data
        predictions = model.transform(test)
        return model, predictions

    def to_submission_format(self, predictions, index_files: dict):
        id_indices = {}
        id_columns = ["tweet_id", "engaging_user_id"]
        for id_column in id_columns:
            id_indices[id_column] = self.spark.read.csv(path=index_files[id_column], sep="\x01", header=True)
            id_indices[id_column] = id_indices[id_column].withColumn(id_column + "_index", F.col(id_column + "_index").cast(IntegerType()))
            predictions = predictions.join(id_indices[id_column], [id_column + "_index"])\
                .drop(id_column + "_index")
        predictions = predictions.withColumn("prediction_bool", F.when(F.col("prediction") >= 0.5, 1).otherwise(0))
        
        return predictions

    def write_to_csv(self, df, fileName: str, single_file: bool = True):
        self.preproc.write_to_csv(df, fileName, single_file)