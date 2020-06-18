import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import when
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
        self.train = self.read_file(train_file, is_preprocessed, n_users, n_tweets)

    def read_test(self, test_file: str, is_preprocessed=True, n_users: int = 0, n_tweets: int = 0):
        self.test = self.read_file(test_file, is_preprocessed, n_users, n_tweets)

    def read_file(self, file_path: str, is_preprocessed: bool = True, n_users: int = 0, n_tweets: int = 0):
        if is_preprocessed:
            return self.preproc.read_preprocessed(file_path)
        return self.preproc.read_raw(file_path, n_users, n_tweets)

    def index_ids(self, df_train, df_test, index_files: dict = None):
        self.id_indices = {}
        id_columns = ["tweet_id", "engaging_user_id"]
        for id_column in id_columns:
            if index_files is None or id_column not in index_files:
                df_id = df_train.union(df_test).select(id_column).distinct()
                self.id_indices[id_column] = self.preproc.get_id_indices(df_id, id_column)
            else:
                self.id_indices[id_column] = self.spark.read.csv(path=index_files[id_column], sep="\x01", header=True)
                self.id_indices[id_column] = self.id_indices[id_column].withColumn(id_column + "_index", F.col(id_column + "_index").cast(LongType()))
            
            df_train = df_train.join(self.id_indices[id_column], [id_column]).drop(id_column)
            df_test = df_test.join(self.id_indices[id_column], [id_column]).drop(id_column)
        return df_train, df_test

    def build_full_matrix(self, df_train, df_test):
        user_ids = df_train.union(df_test).select("engaging_user_id_index").distinct()
        tweet_ids = df_train.union(df_test).select("tweet_id_index").distinct()
        cross_joined = user_ids.join(tweet_ids)
        self.full_train = cross_joined.join(df_train, ["engaging_user_id_index", "tweet_id_index"], how="left")

    def train_test_split(self, df, train_split=0.8, neg_ratio=1):
        neg = self.get_negatives(df)
        neg_train, neg_test = neg.randomSplit([train_split, 1 - train_split])
        
        self.train_sets = {}
        self.test_sets = {}
        for engagement in self.ENGAGEMENTS():
            pos = self.get_positives(df, engagement)
            pos_train, pos_test = pos.randomSplit([train_split, 1 - train_split])
            
            self.train_sets[engagement] = pos_train.union(neg_train)
            self.test_sets[engagement] = pos_test.union(neg_test.limit(int(pos_test.count() * neg_ratio)))

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

    def train_predict(self, df_train, df_test, engagement):
        model, predictions = self.fit_transform(df_train, df_test, engagement)
        return predictions

    def fit_transform(self, training, test, engagement):
        # Build the recommendation model using ALS on the training data
        # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        als = ALS(maxIter=5, regParam=0.01, userCol="engaging_user_id_index", itemCol="tweet_id_index", ratingCol=engagement,
                coldStartStrategy="drop", implicitPrefs=True)
        model = als.fit(training)

        # Evaluate the model by computing the RMSE on the test data
        predictions = model.transform(test)
        return model, predictions

    def write_to_csv(self, df, fileName: str, single_file: bool = True):
        self.preproc.write_to_csv(df, fileName, single_file)