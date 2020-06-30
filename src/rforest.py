
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
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#spark = SparkSession.builder.appName("ChiSquareSpark").getOrCreate()
spark = SparkSession \
    .builder \
    .appName("dic-recsys") \
    .getOrCreate()
sc = spark.sparkContext

#train = "///tmp/traintweet_1000.tsv"
#testfile = "///user/e11920598/test_1000.tsv"
# the full train file has 121.386.431 lines
#train = "///user/pknees/RSC20/training.tsv"
# the full test file has 12.434.838 lines
#testfile = "///user/pknees/RSC20/val.tsv"

### for ensemble:
train = "///tmp/supersecret_train40k_bootstrap.tsv"
#train = "///tmp/traintweet_1000.tsv"
testfile = "///tmp/supersecret_test5k_bootstrap.tsv"
valfile = "///tmp/supersecret_ensembletrain5k_bootstrap.tsv"

do_crossval = True

preproc = twitter_preproc(spark, sc, train, testFile=testfile, valFile=valfile)

traindata = preproc.getDF()
testdata = preproc.getTestDF()
valdata = preproc.getValDF()

preds = ["like", "retweet", "reply", "retweet_comment"]

if not do_crossval:
    for feat in preds:
        # train a random forest with default vals...
        rf = RandomForestClassifier(labelCol=feat, featuresCol="all_features", numTrees=20, maxDepth=5, seed=42)
        model = rf.fit(traindata)
        pred = model.transform(testdata)
        pred_out = pred.select("tweet_id","engaging_user_id","probability","prediction")
        pd.DataFrame(pred_out.collect(),
                     columns=["tweet_id","engaging_user_id","probability","prediction"])\
            .to_csv("../output/" + feat + "_out.csv", sep=",")

if do_crossval:
    for feat in preds:
        # train a random forest with the best parameters from a grid search...
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol=feat)
        evaluator.setMetricName('areaUnderPR') # optimisiing for area under Precision Recall curve - same as used for the recsys challenge
        rf = RandomForestClassifier(labelCol=feat, featuresCol="all_features", seed=42)
        paramGrid = (ParamGridBuilder()
                     .addGrid(rf.numTrees, [20, 50])
                     .addGrid(rf.maxDepth, [5, 10, 25])
                     .addGrid(rf.minInstancesPerNode, [1, 3])
                     .build())
        crossval = CrossValidator(estimator=rf,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5,
                                  seed=42)
        cvModel = crossval.fit(traindata)
        pred = cvModel.transform(testdata)
        pred_out = pred.select("tweet_id","engaging_user_id","probability","prediction")
        pd.DataFrame(pred_out.collect(),
                     columns=["tweet_id","engaging_user_id","probability","prediction"])\
            .to_csv("../output/" + "rf_" + feat + "_out_test.csv", sep=",")
        
        if 'valfile' in locals():
            predval = cvModel.transform(valdata)
            predval_out = predval.select("tweet_id","engaging_user_id","probability","prediction")
            pd.DataFrame(predval_out.collect(),
                     columns=["tweet_id","engaging_user_id","probability","prediction"])\
                .to_csv("../output/" + "rf_" + feat + "_out_ensembletrain.csv", sep=",")
