import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import when
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from IPython.display import display

import matrix_factorization
import importlib
importlib.reload(matrix_factorization)
from matrix_factorization import *

#.config("spark.executor.memoryOverhead", "64g")\
spark = SparkSession\
    .builder\
    .appName("val-save-models")\
    .config("spark.executor.heartbeatInterval", "60s")\
    .config("spark.executor.memory", "32g")\
    .config("spark.driver.memory", "32g")\
    .config("spark.driver.maxResultSize", "64g")\
    .config("spark.sql.crossJoin.enabled", True)\
    .getOrCreate()
sc = spark.sparkContext

mf = matrix_factorization(spark, sc)
mf.read_train("val_indexed/full_train_mf_indexed.tsv", is_preprocessed=True)
mf.full_train = mf.train
mf.read_train("val_indexed/train_mf_indexed.tsv", is_preprocessed=True)
mf.read_val("val_indexed/val_mf_indexed.tsv", is_preprocessed=True)

#mf.train, mf.test = mf.index_ids(mf.train, mf.test)#, index_files={"tweet_id": "tweet_id_indices.tsv", "engaging_user_id": "engaging_user_id_indices.tsv"})

#mf.write_to_csv(mf.id_indices["tweet_id"], "tweet_id_indices")
#mf.write_to_csv(mf.id_indices["engaging_user_id"], "engaging_user_id_indices")

#mf.write_to_csv(mf.train, "train_mf_indexed")
#mf.write_to_csv(mf.test, "test_mf_indexed")

#mf.build_full_matrix(mf.train, mf.test)
#mf.write_to_csv(mf.full_train, "full_train_mf_indexed")

for engagement in mf.ENGAGEMENTS():
    if engagement != "retweet":
        continue
    rank = 30
    path = "model_{}_rank_{}".format(engagement, rank)
    
    model, predictions = mf.train_predict(mf.full_train, mf.val, engagement, rank=rank)
    model.save(path)
    mf.write_to_csv(predictions.select("tweet_id_index", "engaging_user_id_index", "prediction"), engagement + "_raw_predictions")

"""
    submission = mf.to_submission_format(predictions, index_files={"tweet_id": "val_indexed/tweet_id_indices.tsv", "engaging_user_id": "val_indexed/engaging_user_id_indices.tsv"})
    mf.write_to_csv(submission.select("tweet_id", "engaging_user_id", "prediction_bool"), engagement + "_predictions")
"""