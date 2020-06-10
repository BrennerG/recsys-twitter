from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import * 
from pyspark.ml.feature import RegexTokenizer, OneHotEncoderEstimator, StringIndexer

class twitter_preproc:
    #CHANGED: Constructor MF:bool=True/False ---> method:str="MF"/"CB"/""
    def __init__(self, spark:SparkSession, sc:SparkContext, inputFile:str, seed:int=123, method:str=""):
        self.sc = sc
        #inputRDD = sc.textFile(inputFile)
        #self.inputData = spark.read.option("sep", "\x01").csv(inputFile)
        SCHEMA = StructType([
                StructField("text_tokens", StringType()),
                StructField("hashtags", StringType()),
                StructField("tweet_id", StringType()),
                StructField("present_media", StringType()),
                StructField("present_links", StringType()),
                StructField("present_domains", StringType()),
                StructField("tweet_type", StringType()),
                StructField("language", StringType()),
                StructField("tweet_timestamp", LongType()),
                StructField("engaged_with_user_id", StringType()),
                StructField("engaged_with_user_follower_count", LongType()),
                StructField("engaged_with_user_following_count", LongType()),
                StructField("engaged_with_user_is_verified", BooleanType()),
                StructField("engaged_with_user_account_creation", LongType()),
                StructField("engaging_user_id", StringType()),
                StructField("engaging_user_follower_count", LongType()),
                StructField("engaging_user_following_count", LongType()),
                StructField("engaging_user_is_verified", BooleanType()),
                StructField("engaging_user_account_creation", LongType()),
                StructField("engaged_follows_engaging", BooleanType()),
                StructField("reply_timestamp", LongType()),
                StructField("retweet_timestamp", LongType()),
                StructField("retweet_with_comment_timestamp", LongType()),
                StructField("like_timestamp", LongType())       
                                ])
        self.inputData = spark.read.csv(path=inputFile, sep="\x01", header=False, schema=SCHEMA)
        
        if(method==""):
            self._preprocess(seed)
        elif(method=="MF"):
            self._preprocessMF()
        elif(method=="CB"):
            self._preprocessCB()
            
        #self.inputData = spark.createDataFrame(inputRDD, sep="\x01", schema=SCHEMA)    
    
    def getDF(self):
        return self.outputDF
    
    def _preprocessCB(self):
        outputDF = self.inputData
        
        self.outputDF = outputDF.select(["tweet_id","engaging_user_id","engaged_with_user_id",
                                    "retweet_timestamp","reply_timestamp",
                                    "retweet_with_comment_timestamp","like_timestamp","text_tokens"])
            
    def _preprocessMF(self):
        outputDF = self.inputData
        
        self.outputDF = outputDF.select(["tweet_id","engaging_user_id","engaged_with_user_id",
                                    "retweet_timestamp","reply_timestamp",
                                    "retweet_with_comment_timestamp","like_timestamp"])
    
    def _preprocess(self, seed):
        
        outputDF = self.inputData
        
        # Drop unnecessary cols
        ### drop ids for classification
        outputDF = outputDF.drop("tweet_id").drop("engaged_user_id").drop("engaged_with_user_id")\
                    .drop("present_links").drop("present_domains")
        
        # Split the text tokens to valid format
        regexTokenizer = RegexTokenizer(inputCol="text_tokens",outputCol="vector", pattern="\t")
        outputDF = regexTokenizer.transform(outputDF)
        outputDF = outputDF.drop("text_tokens").withColumnRenamed("vector", "text_tokens")
        
        regexTokenizer = RegexTokenizer(inputCol="present_media", outputCol="media_list")
        outputDF = regexTokenizer.transform(outputDF.fillna("none", subset=["present_media"]))
        outputDF = outputDF.drop("present_media").withColumnRenamed("media_list", "present_media")
        outputDF = outputDF.withColumn("present_media2", outputDF["present_media"].cast(StringType()))
        outputDF = outputDF.drop("present_media").withColumnRenamed("present_media2", "present_media")

        # OneHotEncode tweet_type
        ## TODO: user_id, engaged_user_id, ...
        indexer = StringIndexer(inputCol="tweet_type", outputCol="tweet_type_id")
        outputDF = indexer.fit(outputDF).transform(outputDF)
        indexer = StringIndexer(inputCol="present_media", outputCol="present_media_id")
        outputDF = indexer.fit(outputDF).transform(outputDF)
        indexer = StringIndexer(inputCol="language", outputCol="language_id")
        outputDF = indexer.fit(outputDF).transform(outputDF)
        
        # onehot
        encoder = OneHotEncoderEstimator(inputCols=["tweet_type_id", "present_media_id", "language_id"],
                                         outputCols=["tweet_type_onehot", "present_media_onehot", "language_onehot"])
        model = encoder.fit(outputDF)
        outputDF = model.transform(outputDF)
        
        # for explainability safe this
        self.explainOneHotDF = outputDF.select("tweet_type", "tweet_type_id", "tweet_type_onehot",
                                              "present_media", "present_media_id", "present_media_onehot",
                                               "language", "language_id", "language_onehot"
                                              )
        # make label columns binary
        outputDF = outputDF.withColumn("like", when(outputDF["like_timestamp"].isNull(), 0).otherwise(1))
        outputDF = outputDF.withColumn("retweet", when(outputDF["retweet_timestamp"].isNull(), 0).otherwise(1))
        outputDF = outputDF.withColumn("reply", when(outputDF["reply_timestamp"].isNull(), 0).otherwise(1))
        outputDF = outputDF.withColumn("retweet_comment", when(outputDF["retweet_with_comment_timestamp"].isNull(), 0).otherwise(1))
        
        # drop intermediate columns
        outputDF = outputDF.drop(*["like_timestamp","retweet_timestamp","reply_timestamp",
                                  "retweet_with_comment_timestamp","tweet_type","tweet_type_id",
                                 "language","language_id","present_media","present_media_id"])
        
        # TODO: Train/Test split and Scaling
        
        # might not need
        # transform boolean to 0-1 column... first one has to change the type in the schema though 
        #data = data.select("engaging_user_is_verified", "engaged_with_user_is_verified", "engaged_follows_engaging")\
        #    .replace(["false","true"], ["0","1"]).show()
        
        
        self.outputDF = outputDF
        
    '''
        returns small dataframe that explains the values of the oneHotEncoder step, this might be needed
        for mapping the encodings back to the original values
    '''    
    def explainOneHot(self):
        return self.explainOneHotDF
