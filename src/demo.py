

from twitter_preproc import twitter_preproc
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

#spark = SparkSession.builder.appName("ChiSquareSpark").getOrCreate()
spark = SparkSession \
    .builder \
    .appName("Pipeline") \
    .getOrCreate()
sc = spark.sparkContext


# sample file with 1000 tweets for checking the pipeline
train = "///user/e11920598/traintweet_1000.tsv"

preproc = twitter_preproc(spark, sc, train)
print(preproc.getDF().show(5))


sc.stop()
