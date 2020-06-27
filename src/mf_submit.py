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
    .appName("mf-val-submission")\
    .config("spark.executor.heartbeatInterval", "60s")\
    .config("spark.executor.memory", "32g")\
    .config("spark.driver.memory", "32g")\
    .config("spark.driver.maxResultSize", "64g")\
    .config("spark.sql.crossJoin.enabled", True)\
    .getOrCreate()
sc = spark.sparkContext

mf = matrix_factorization(spark, sc)
mf.read_train("challenge/val_indexed/train_mf_indexed.tsv", is_preprocessed=True)
mf.read_val("challenge/val_indexed/val_mf_indexed.tsv", is_preprocessed=True)

index_files = {"tweet_id": "challenge/val_indexed/tweet_id_indices.tsv", "engaging_user_id": "challenge/val_indexed/engaging_user_id_indices.tsv"}
#mf.train, mf.val = mf.index_ids(mf.train, mf.val, index_files=index_files)

#mf.write_to_csv(mf.id_indices["tweet_id"], "tweet_id_indices")
#mf.write_to_csv(mf.id_indices["engaging_user_id"], "engaging_user_id_indices")

#mf.write_to_csv(mf.train, "challenge/val_indexed/train_mf_indexed")
#mf.write_to_csv(mf.val, "challenge/val_indexed/val_mf_indexed")

full_train = mf.build_full_matrix(mf.train, mf.val, cross_join=False)
mf.write_to_csv(full_train, "challenge/val_indexed/full_train_mf_indexed")

for engagement in mf.ENGAGEMENTS():
    alpha = 10.0
    rank = 50
    threshold = 0.001
    model_path = "challenge/val_indexed/model_{}_a{}_r{}".format(engagement, int(alpha), rank)
    
    model, predictions = mf.train_predict(full_train, mf.val, engagement, rank, alpha)
    model.save(model_path)
    mf.write_to_csv(predictions, "challenge/val_indexed/raw_predictions_{}_a{}_r{}".format(engagement, int(alpha), rank))

    submission = mf.to_submission_format(predictions, index_files, threshold)
    mf.write_to_csv(submission.select("tweet_id", "engaging_user_id", "prediction_bool"), "challenge/val_indexed/submission_" + engagement)