import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import when
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
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
    .appName("matrix-factorization")\
    .config("spark.executor.heartbeatInterval", "60s")\
    .config("spark.executor.memory", "8g")\
    .config("spark.driver.memory", "8g")\
    .config("spark.driver.maxResultSize", "64g")\
    .config("spark.sql.crossJoin.enabled", True)\
    .getOrCreate()
sc = spark.sparkContext

mf = matrix_factorization(spark, sc)
mf.read_train("train_mf_indexed.tsv", is_preprocessed=True)
mf.read_test("test_mf_indexed.tsv", is_preprocessed=True)

#mf.train, mf.test = mf.index_ids(mf.train, mf.test, index_files={"tweet_id": "tweet_id_indices.tsv", "engaging_user_id": "engaging_user_id_indices.tsv"})

#mf.write_to_csv(mf.id_indices["tweet_id"], "tweet_id_indices")
#mf.write_to_csv(mf.id_indices["engaging_user_id"], "engaging_user_id_indices")

#mf.write_to_csv(mf.train, "train_mf_indexed")
#mf.write_to_csv(mf.test, "test_mf_indexed")

mf.build_full_matrix(mf.train, mf.test)
#mf.write_to_csv(mf.full_train, "full_train_mf_indexed")

for engagement in mf.ENGAGEMENTS():
    predictions = mf.train_predict(mf.full_train, mf.test, engagement)
    mf.write_to_csv(predictions, engagement + "_predictions")