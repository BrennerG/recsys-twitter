import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

import train_test_split
import importlib
importlib.reload(train_test_split)
from train_test_split import *

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

file = "hdfs://master:9000/user/pknees/RSC20/training.tsv"
split = train_test_split(spark, sc)
split.read_raw(file, full=False)

split.write_to_csv(split.df, "training_mf_indexed")