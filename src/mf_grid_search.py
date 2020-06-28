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
import matplotlib.pyplot as plt

from matrix_factorization import *
from recsys_evaluator import *

def file_exists(file_name):
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    return fs.exists(sc._jvm.org.apache.hadoop.fs.Path(file_name))

#.config("spark.executor.memoryOverhead", "64g")\
spark = SparkSession\
    .builder\
    .appName("mf-grid-search")\
    .config("spark.executor.heartbeatInterval", "60s")\
    .config("spark.executor.memory", "32g")\
    .config("spark.driver.memory", "32g")\
    .config("spark.driver.maxResultSize", "64g")\
    .config("spark.sql.crossJoin.enabled", True)\
    .getOrCreate()
sc = spark.sparkContext

mf = matrix_factorization(spark, sc)
mf.read_train("test_1000/train.tsv", is_preprocessed=True)
mf.read_test("test_1000/test.tsv", is_preprocessed=True)

eval_values = []
thresholds = [0.001, 0.01, 0.1, 0.3, 0.5]
for engagement in mf.ENGAGEMENTS():
    for alpha in [1.0, 10.0]:
        for rank in [10, 50]:
            model_path = "test_1000/model_{}_a{}_r{}".format(engagement, int(alpha), rank)
            if file_exists(model_path):
                model = ALSModel.load(model_path)
                predictions = model.transform(mf.test)
                areasUnderPR, log_loss = mf.evaluate(predictions, engagement, thresholds)
            else:
                model, predictions, areasUnderPR, log_loss = mf.train_evaluate(mf.train, mf.test, engagement, rank=rank, alpha=alpha, thresholds=thresholds)
                model.save()
            
            eval_value = {"engagement": engagement, "alpha": alpha, "rank": rank, "log_loss": log_loss}
            for i, threshold in enumerate(thresholds):
                eval_value["areaUnderPR-{}".format(threshold)] = areasUnderPR[i]
            eval_values.append(eval_value)
            
            eval_df = pd.DataFrame(eval_values)
            eval_df.to_csv("../grid-search/mf/mf_eval_values.csv", index=False)
            
            pd_df = pd.DataFrame(predictions.collect(), columns=predictions.columns)
            normalized = (pd_df["prediction"] - pd_df["prediction"].min()) / (pd_df["prediction"].max() - pd_df["prediction"].min())

            pd_df["prediction"].hist(bins=50)
            plt.savefig("../grid-search/mf/plots/mf_predictions_{}_a{}_r{}.png".format(engagement, int(alpha), rank))

            normalized.hist(bins=50)
            plt.savefig("../grid-search/mf/plots/mf_predictions_normalized_{}_a{}_r{}.png".format(engagement, int(alpha), rank))