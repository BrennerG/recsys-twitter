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

def challenge_submission(mf):
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
        
def predict_for_ensemble(mf):
    # dirty hack: When calling mf.read_test no indices will be deleted
    mf.read_test("ensemble/full_train_mf_indexed.tsv", is_preprocessed=True)
    df_train = mf.test
    mf.read_test("/tmp/supersecret_ensembletrain5k_bootstrap.tsv", is_preprocessed=False)
    df_ensembletrain = mf.test
    mf.read_test("/tmp/supersecret_test5k_bootstrap.tsv", is_preprocessed=False)
    df_test = mf.test

    index_files = {"tweet_id": "challenge/val_indexed/tweet_id_indices.tsv", "engaging_user_id": "challenge/val_indexed/engaging_user_id_indices.tsv"}
    #df_train, _ = mf.index_ids(df_train, index_files=index_files)
    df_ensembletrain, _ = mf.index_ids(df_ensembletrain, index_files=index_files)
    df_test, _ = mf.index_ids(df_test, index_files=index_files)

    full_train = df_train
    #full_train = mf.build_full_matrix(df_train, df_ensembletrain.unionByName(df_test), cross_join=True)
    #mf.write_to_csv(full_train, "ensemble/full_train_mf_indexed")

    for engagement in mf.ENGAGEMENTS():
        alpha = 10.0
        rank = 50
        model_path = "ensemble/model_{}_a{}_r{}".format(engagement, int(alpha), rank)

        model, train_predictions = mf.train_predict(full_train, df_ensembletrain, engagement, rank, alpha)
        test_predictions = model.transform(df_test)
        model.save(model_path)
        train_submission = mf.to_submission_format(train_predictions, index_files)
        test_submission = mf.to_submission_format(test_predictions, index_files)
        mf.write_to_csv(train_submission.select("tweet_id", "engaging_user_id", "prediction"), "ensemble/train_prediction_{}_a{}_r{}".format(engagement, int(alpha), rank))
        mf.write_to_csv(test_submission.select("tweet_id", "engaging_user_id", "prediction"), "ensemble/test_prediction_{}_a{}_r{}".format(engagement, int(alpha), rank))


#.config("spark.executor.memoryOverhead", "64g")\
spark = SparkSession\
    .builder\
    .appName("mf-ensemble-prediction")\
    .config("spark.executor.heartbeatInterval", "60s")\
    .config("spark.executor.memory", "32g")\
    .config("spark.driver.memory", "32g")\
    .config("spark.driver.maxResultSize", "64g")\
    .config("spark.sql.crossJoin.enabled", True)\
    .getOrCreate()
sc = spark.sparkContext

mf = matrix_factorization(spark, sc)

# challenge_submission(mf)
predict_for_ensemble(mf)