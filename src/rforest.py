
import twitter_preproc
import importlib
importlib.reload(twitter_preproc)
from twitter_preproc import *
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from twitter_preproc import twitter_preproc

#spark = SparkSession.builder.appName("ChiSquareSpark").getOrCreate()
spark = SparkSession \
    .builder \
    .appName("dic-recsys") \
    .getOrCreate()
sc = spark.sparkContext

#train = "///tmp/traintweet_1000.tsv"
#testfile = "///user/e11920598/test_1000.tsv"
# the full train file has 121.386.431 lines
train = "///user/pknees/RSC20/training.tsv"
# the full test file has 12.434.838 lines
testfile = "///user/pknees/RSC20/test.tsv"

preproc = twitter_preproc(spark, sc, train, testFile=testfile)

traindata = preproc.getDF()
testdata = preproc.getTestDF()

preds = ["like", "retweet", "reply", "retweet_comment"]

for feat in preds:
    # train a random forest with default vals...
    rf = RandomForestClassifier(labelCol=feat, featuresCol="all_features", numTrees=20, maxDepth=5, seed=42)
    model = rf.fit(traindata)
    pred = model.transform(testdata)
    pred_out = pred.select("tweet_id","engaging_user_id","probability","prediction")
    pd.DataFrame(pred_out.collect(),
                 columns=["tweet_id","engaging_user_id","probability","prediction"])\
        .to_csv("../output/" + feat + "_out.csv", sep=",")
