import pandas as pd
from typing import List, Dict

from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from IPython.display import display

from recsys_evaluator import *

class ensemble:

    def __init__(self, spark: SparkSession, sc: SparkContext):
        self.spark = spark
        self.sc = sc
        self.evaluator = recsys_evaluator(self.spark, self.sc)
        self.INDEX_COLS = ["engaging_user_id", "tweet_id"]

    def train(self, prediction_files: Dict[str, str], prediction_schemas: Dict[str, StructType], label_file: str, engagement: str):
        """
        :param prediction_files: dict of model names with hdfs paths containg their predictions as tsv, including the columns "engaging_user_id", "tweet_id", "prediction" or specified in schema
        :param label_file: hdfs path containing the labels as tsv, including "engaging_user_id", "tweet_id", `engagement`
        """
        predictions_per_model = self.read_predictions(prediction_files, prediction_schemas)
        predictions = self.join_and_vectorize_predictions(predictions_per_model)
        labels = self.read_labels(label_file, engagement)
        
        train_df = predictions.join(labels, self.INDEX_COLS)
        lr_model = LinearRegression(featuresCol="features", labelCol="label", elasticNetParam=0.0).fit(train_df)
        return lr_model

    def test_evaluate(self, lr_model: LinearRegressionModel, prediction_files: Dict[str, str], prediction_schemas: Dict[str, StructType], 
        label_file: str, engagement: str, thresholds: List[float]):
        """
        :param prediction_files: dict of model names with hdfs paths containg their predictions as tsv, including the columns "engaging_user_id", "tweet_id", "prediction"
        :param label_file: hdfs path containing the labels as tsv, including "engaging_user_id", "tweet_id", `engagement`
        """
        predictions_per_model = self.read_predictions(prediction_files, prediction_schemas)
        predictions = self.join_and_vectorize_predictions(predictions_per_model)
        labels = self.read_labels(label_file, engagement)
        
        test_df = predictions.join(labels, self.INDEX_COLS)
        pred_df = lr_model.transform(test_df)

        eval_values = []
        for model_name in predictions_per_model:
            model_prediction = predictions_per_model[model_name]
            eval_row = self.evaluate(model_prediction, labels, model_name, thresholds)
            eval_row["model_name"] = model_name
            eval_values.append(eval_row)
        
        eval_row = self.evaluate(pred_df, labels, "prediction", thresholds)
        eval_row["model_name"] = "Ensemble"
        eval_values.append(eval_row)
        
        return pd.DataFrame(eval_values)

    def read_predictions(self, prediction_files, prediction_schemas):
        """
        :return a dict with one dataframe per model, each with the columns "engaging_user_id" (string), "tweet_id" (string), `model_name` (double)
        """
        predictions_per_model = {}
        for model_name in prediction_files:
            sep = "\x01" if model_name == "mf" else ","
            schema = None if model_name not in prediction_schemas else prediction_schemas[model_name]
            df = self.spark.read.csv(path=prediction_files[model_name], sep=sep, header=True) if schema is None else self.spark.read.csv(path=prediction_files[model_name], sep=sep, schema=schema)
            if len(set(self.INDEX_COLS + ["prediction"]) & set(df.columns)) != 3:
                raise Exception("Prediction file has missing columns: {}".format(set(self.INDEX_COLS + ["prediction"]) - set(df.columns)))
            
            predictions_per_model[model_name] = df.withColumn(model_name, F.col("prediction").cast(DoubleType()))\
                .select(self.INDEX_COLS + [model_name])

        return predictions_per_model

    def join_and_vectorize_predictions(self, predictions_per_model: Dict[str, DataFrame]):
        """
        :return a dataframe with the columns "engaging_user_id" (string), "tweet_id" (string), "features" (vector)
        """
        predictions = None
        for model_name in predictions_per_model:
            df = predictions_per_model[model_name]
            if predictions is None:
                predictions = df
            else:
                predictions = predictions.join(df, self.INDEX_COLS)
        
        vector_assembler = VectorAssembler(inputCols=list(predictions_per_model.keys()), outputCol="vectors")
        min_max_scaler = MinMaxScaler(inputCol="vectors", outputCol="features")
        pipeline = Pipeline(stages=[vector_assembler, min_max_scaler])
        return pipeline\
            .fit(predictions)\
            .transform(predictions)\
            .select(self.INDEX_COLS + ["features"])

    def read_labels(self, label_file, engagement):
        """
        :return a dataframe with the columns "engaging_user_id" (string), "tweet_id" (string), "label" (byte)
        """
        labels = self.spark.read.csv(path=label_file, sep="\x01", header=True)
        if len(set(self.INDEX_COLS + [engagement]) & set(labels.columns)) != 3:
                raise Exception("Label file has missing columns: {}".format(set(self.INDEX_COLS + [engagement]) - set(labels.columns)))
        return labels.withColumn("label", F.col(engagement).cast(ByteType()))

    def evaluate(self, predictions: DataFrame, labels: DataFrame, p_col: str, thresholds: List[float]):
        """
        :return dict containing the evaluation values "log_loss" and "areaUnderPR" for every threshold
        """
        pred_labels = predictions
        if "label" not in pred_labels.columns:
            pred_labels = pred_labels.join(labels, self.INDEX_COLS)
        
        areasUnderPR = self.evaluator.area_under_pr_curve(pred_labels, e_col="label", p_col=p_col, thresholds=thresholds)
        log_loss = self.evaluator.log_loss(pred_labels, e_col="label", p_col=p_col)

        eval_row = {"log_loss": log_loss}
        for i, threshold in enumerate(thresholds):
            eval_row["areaUnderPR-{}".format(threshold)] = areasUnderPR[i]
        return eval_row