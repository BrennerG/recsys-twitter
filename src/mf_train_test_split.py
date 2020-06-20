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
    .appName("train-test-split")\
    .config("spark.executor.heartbeatInterval", "60s")\
    .config("spark.executor.memory", "32g")\
    .config("spark.driver.memory", "32g")\
    .config("spark.driver.maxResultSize", "64g")\
    .config("spark.sql.crossJoin.enabled", True)\
    .getOrCreate()
sc = spark.sparkContext

mf = matrix_factorization(spark, sc)
mf.read_train("///user/pknees/RSC20/training.tsv", is_preprocessed=False)

df_train, _ = mf.index_ids(mf.train)
mf.write_to_csv(df_train, "full_train_indexed")

#df_train = df_train.limit(10 ** 6)
n_train = df_train.count()
n_test = 1000
df_train, df_test = mf.train_test_split(df_train, train_split=((n_train - n_test) / n_train))

mf.write_to_csv(df_train, "test_1000/train")
mf.write_to_csv(df_test, "test_1000/test")