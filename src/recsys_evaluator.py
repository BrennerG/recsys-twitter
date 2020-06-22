import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import when
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from IPython.display import display
import matplotlib.pyplot as plt

class recsys_evaluator:

    def __init__(self, spark: SparkSession, sc: SparkContext):
        self.spark = spark
        self.sc = sc

    def add_binary_prediction_col(self, df, p_col: str, threshold: float):
        return df.withColumn(p_col + "_binary", F.when(F.col(p_col) >= threshold, 1.0).otherwise(0.0))

    def area_under_pr_curve(self, df, e_col: str, p_col: str, thresholds: list):
        """
        :param e_col: Column name of the true engagement
        :param p_col: Column name of the predicted engagement
        :param thresholds: List of float, containing different thresholds, map predictions above to 1 (positive engagement)
        """
        areas = []
        for threshold in thresholds:
            df_binary = self.add_binary_prediction_col(df, p_col, threshold)
            evaluator = BinaryClassificationEvaluator(rawPredictionCol=p_col + "_binary", labelCol=e_col, metricName="areaUnderPR")
            areas.append(evaluator.evaluate(df_binary))
        return areas

    def log_loss(self, df, e_col: str, p_col: str):
        """
        :param e_col: Column name of the true engagement
        :param p_col: Column name of the predicted engagement
        """
        y1, y0 = df[e_col], 1 - df[e_col]
        p1, p0 = df[p_col], 1 - df[p_col]
        # negative log-likelihood
        nll = -(y1 * F.log(p1) + y0 * F.log(p0))
        return df.agg(F.mean(nll)).collect()[0][0]