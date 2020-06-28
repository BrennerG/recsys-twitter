import pandas as pd
from typing import List, Dict

from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from IPython.display import display

class ensemble:

    def __init__(self, spark: SparkSession, sc: SparkContext):
        self.spark = spark
        self.sc = sc
        self.INDEX_COLS = ["engaging_user_id", "tweet_id"]

    def train(self, prediction_files: Dict[str, str], label_file: str):
        """
        :param prediction_files: dict of model names with hdfs paths containg their predictions as tsv, including the columns "engaging_user_id", "tweet_id", "prediction"
        :param label_file: hdfs path containing the labels as tsv, including "engaging_user_id", "tweet_id", "label"
        """
        predictions = self.read_predictions(prediction_files)
        labels = self.read_labels(label_file)
        
        train_df = predictions.join(labels, self.INDEX_COLS)
        lr_model = LinearRegression(featuresCol="features", labelCol="label", elasticNetParam=0.0).fit(train_df)
        print("RMSE: {}".format(lr_model.summary.rootMeanSquaredError))
        return lr_model

    def test_evaluate(self, lr_model: LinearRegressionModel, prediction_files: Dict[str, str], label_file: str):
        """
        :param prediction_files: dict of model names with hdfs paths containg their predictions as tsv, including the columns "engaging_user_id", "tweet_id", "prediction"
        :param label_file: hdfs path containing the labels as tsv, including "engaging_user_id", "tweet_id", "label"
        """
        predictions = self.read_predictions(prediction_files)
        labels = self.read_labels(label_file)
        
        test_df = predictions.join(labels, self.INDEX_COLS)
        pred_df = lr_model.transform(test_df)
        return pred_df

    def read_predictions(self, prediction_files):
        """
        :return a dataframe with the columns "engaging_user_id" (string), "tweet_id" (string), "features" (vector)
        """
        predictions = None
        for model_name in prediction_files:
            df = self.spark.read.csv(path=prediction_files[model_name], sep="\x01", header=True)
            if len(set(self.INDEX_COLS + ["prediction"]) & set(df.columns)) != 3:
                raise Exception("Prediction file has missing columns: {}".format(set(self.INDEX_COLS + ["prediction"]) - set(df.columns)))
            
            df = df.withColumn(model_name, F.col("prediction").cast(DoubleType()))\
                .select(self.INDEX_COLS + [model_name])

            if predictions is None:
                predictions = df
            else:
                predictions = predictions.join(df, self.INDEX_COLS)
        
        return VectorAssembler(inputCols=list(prediction_files.keys()), outputCol="features")\
            .transform(predictions)\
            .select(self.INDEX_COLS + ["features"])

    def read_labels(self, label_file):
        """
        :return a dataframe with the columns "engaging_user_id" (string), "tweet_id" (string), "label" (byte)
        """
        labels = self.spark.read.csv(path=label_file, sep="\x01", header=True)
        if len(set(self.INDEX_COLS + ["label"]) & set(labels.columns)) != 3:
                raise Exception("Label file has missing columns: {}".format(set(self.INDEX_COLS + ["label"]) - set(labels.columns)))
        return labels.withColumn("label", F.col("label").cast(ByteType()))