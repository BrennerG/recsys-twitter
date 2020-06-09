from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *

class twitter_preproc:
    def __init__(self, spark:SparkSession, sc:SparkContext, inputFile:str, seed:int=123, MF:bool=False):
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
        if MF:
            self._preprocessMF()
        else:
            self._preprocess(seed)
        #self.inputData = spark.createDataFrame(inputRDD, sep="\x01", schema=SCHEMA)    
    
    def getDF(self):
        return self.outputDF
    
    def _preprocessMF(self):
        outputDF = self.inputData
        
        self.outputDF = outputDF.select(["tweet_id","engaging_user_id","engaged_with_user_id",
                                    "retweet_timestamp","reply_timestamp",
                                    "retweet_with_comment_timestamp","like_timestamp"])
        
        
    
    def _preprocess(self, seed):
        from pyspark.ml.feature import RegexTokenizer, OneHotEncoderEstimator, StringIndexer
        
        outputDF = self.inputData
        
        # Drop unnecessary cols
        ### drop ids for classification
        outputDF = outputDF.drop("tweet_id").drop("engaged_user_id").drop("engaged_with_user_id")
        #
        
        # Split the text tokens to valid format
        regexTokenizer = RegexTokenizer(inputCol="text_tokens",outputCol="vector", pattern="\t")
        outputDF = regexTokenizer.transform(outputDF)
        outputDF = outputDF.drop("text_tokens").withColumnRenamed("vector", "text_tokens")
        
        #regexTokenizer = RegexTokenizer(inputCol="present_media", outputCol="media_list")
        #outputDF = regexTokenizer.transform(regexTokenizer)
        #outputDF = outputDF.drop("present_media").withColumnRenamed("vector", "text_tokens")
        
        # OneHotEncode tweet_type
        ## TODO: user_id, engaged_user_id, ...
        indexer = StringIndexer(inputCol="tweet_type", outputCol="tweet_type_id")
        outputDF = indexer.fit(outputDF).transform(outputDF)
        #onehot
        encoder = OneHotEncoderEstimator(inputCols=["tweet_type_id"], outputCols=["tweet_type_onehot"])
        model = encoder.fit(outputDF)
        outputDF = model.transform(outputDF)
        # for explainability safe this
        self.explainOneHotDF = outputDF.select("tweet_type", "tweet_type_id", "tweet_type_onehot")
        
        # tf/idf text_tokens, hashtags, 
        
        
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
