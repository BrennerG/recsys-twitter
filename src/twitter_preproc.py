from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import * 
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import RegexTokenizer, OneHotEncoderEstimator, StringIndexer, MinMaxScaler, VectorAssembler, HashingTF, IDF

class twitter_preproc:
    
    def __init__(self, spark:SparkSession, sc:SparkContext, trainFile:str, testFile:str="", seed:int=123,
                 MF:bool=False, trainsplit:float=0.9):
        
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
        
        self.trainFile = spark.read.csv(path=trainFile, sep="\x01", header=False, schema=SCHEMA)
        
        if MF:
            self._preprocessMF()
        else:
            self._preprocess(trainsplit, seed)
        #self.inputData = spark.createDataFrame(inputRDD, sep="\x01", schema=SCHEMA)    
        
        if testFile:
            self.testFile = spark.read.csv(path=testFile, sep="\x01", header=False, schema=SCHEMA)
            self._preprocessTest()
    
    '''
        get the outputDF of the class, which is the result of the input after all preprocessing steps
    '''
    def getDF(self):
        return self.processedTrainDF
    
    '''
        get the preprocessed testDF
    '''
    def getTestDF(self):
        return self.processedTestDF
    
    '''
        return assembled DF, meaning all columsn from preprocessing steps are merged to one vector 
        (this is need for sparkML). This drops the labels
    '''
    def _assemble(self, df):
        
        cols = df.columns
        cols.remove("like") # remove labels and identifiers
        cols.remove("retweet")
        cols.remove("reply")
        cols.remove("retweet_comment")
        cols.remove("tweet_id")
        cols.remove("engaging_user_id")
        assembler = VectorAssembler(inputCols=cols, outputCol="all_features")
        assembledDF = assembler.transform(df)
        return assembledDF
    
    def _preprocessMF(self):
        outputDF = self.trainFile
        
        self.outputDF = outputDF.select(["tweet_id","engaging_user_id","engaged_with_user_id",
                                    "retweet_timestamp","reply_timestamp",
                                    "retweet_with_comment_timestamp","like_timestamp"])
    
    def _preprocess(self, trainsplit, seed):
        
        outputDF = self.trainFile
        
        # Drop unnecessary cols
        ### drop unused ids for classification
        outputDF = outputDF.drop("engaged_with_user_id").drop("engaged_user_id")\
                    .drop("present_links").drop("present_domains")
        #.drop("tweet_id")
        #.drop("engaging_user_id")
        
        # Split the text tokens to valid format
        textTokenizer = RegexTokenizer(inputCol="text_tokens",outputCol="vector", pattern="\t")
        outputDF = textTokenizer.transform(outputDF)
        hashtagTokenizer = RegexTokenizer(inputCol="hashtags",outputCol="hashtag_tokens", pattern="\t")
        outputDF = hashtagTokenizer.transform(outputDF.fillna("none", subset=["hashtags"]))
        
        #self.tokenizerPipeline = Pipeline(stages=[textTokenizer, hashtagTokenizer])
        #outputDF = self.tokenizerPipeline.fit(outputDF).transform(outputDF)
        
        outputDF = outputDF.drop("text_tokens").withColumnRenamed("vector", "text_tokens")
        outputDF = outputDF.drop("hashtags").withColumnRenamed("hashtag_tokens", "hashtags")
        
        regexTokenizer = RegexTokenizer(inputCol="present_media", outputCol="media_list")
        outputDF = regexTokenizer.transform(outputDF.fillna("none", subset=["present_media"]))
        outputDF = outputDF.drop("present_media").withColumnRenamed("media_list", "present_media")
        outputDF = outputDF.withColumn("present_media2", outputDF["present_media"].cast(StringType()))
        outputDF = outputDF.drop("present_media").withColumnRenamed("present_media2", "present_media")

        # OneHotEncode tweet_type
        ## TODO: user_id, engaged_user_id, ...
        indexerTweetType = StringIndexer(inputCol="tweet_type", outputCol="tweet_type_id", handleInvalid="keep" )
        #outputDF = indexerTweetType.fit(outputDF).transform(outputDF)
        indexerMedia = StringIndexer(inputCol="present_media", outputCol="present_media_id", handleInvalid="keep")
        #outputDF = indexerMedia.fit(outputDF).transform(outputDF)
        indexerLang = StringIndexer(inputCol="language", outputCol="language_id", handleInvalid="keep")
        #outputDF = indexerLang.fit(outputDF).transform(outputDF)
        
        indexerPipeline = Pipeline(stages=[indexerTweetType, indexerMedia, indexerLang]) 
        self.indexerModel = indexerPipeline.fit(outputDF)
        outputDF = self.indexerModel.transform(outputDF)
        
        # onehot
        encoder = OneHotEncoderEstimator(inputCols=["tweet_type_id", "present_media_id", "language_id"],
                                         outputCols=["tweet_type_onehot", "present_media_onehot", "language_onehot"])
        self.encoderModel = encoder.fit(outputDF)
        outputDF = self.encoderModel.transform(outputDF)
        
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
        
        # tf/idf text + hashtags
        ### hashtags
        hashtagsTF = HashingTF(inputCol="hashtags", outputCol="hashtagsTF", numFeatures=2^10)
        #outputDF = hashtagsTF.transform(outputDF)
        hashtagsIDF = IDF(inputCol="hashtagsTF", outputCol="hashtags_idf")
        #outputDF = self.hashtagsIDF.fit(outputDF).transform(outputDF)
        
        textTF = HashingTF(inputCol="text_tokens", outputCol="tweet_text_TF", numFeatures=2^14)
        #outputDF = textTF.transform(outputDF)
        textIDF = IDF(inputCol="tweet_text_TF", outputCol="tweet_text_idf")
        #outputDF = self.textIDF.fit(outputDF).transform(outputDF)
        
        tfidfPipeline = Pipeline(stages=[hashtagsTF, hashtagsIDF, textTF, textIDF])
        self.tfidfModel = tfidfPipeline.fit(outputDF)
        outputDF = self.tfidfModel.transform(outputDF)
        
        outputDF = outputDF.drop(*["hashtags", "hashtagsTF", "text_tokens", "tweet_text_TF"])
        
        # scaling
        '''
        scalerTimestamp = MinMaxScaler(inputCol="tweet_timestamp",
                                       outputCol="tweet_timestamp_scaled")
        scalerEngagedAccountCreation = MinMaxScaler(inputCol="engaged_with_user_account_creation",
                                                   outputCol="engaged_with_user_account_creation_scaled")
        scalerEngagingAccountCreation = MinMaxScaler(inputCol="engaging_user_account_creation",
                                                    outputCol="engaging_user_account_creation_scaled")
        
        scalerEngagedFollowerCount = MinMaxScaler(inputCol="engaged_with_user_follower_count",
                                            outputCol="engaged_with_user_follower_count_scaled")
        scalerEngagedFollowingCount = MinMaxScaler(inputCol="engaged_with_user_following_count",
                                                  outputCol="engaged_with_user_following_count_scaled")
        scalerEngagingFollowerCount = MinMaxScaler(inputCol="engaging_user_follower_count",
                                           outputCol="engaging_user_follower_count_scaled")
        scalerEngagingFollowingCount = MinMaxScaler(inputCol="engaging_user_following_count",
                                                   outputCol="engaging_user_following_count_scaled")
        scalePipeline = Pipeline(stages=[scalerTimestamp, scalerEngagedAccountCreation,
                                         scalerEngagingAccountCreation, scalerEngagedFollowerCount,
                                        scalerEngagedFollowingCount, scalerEngagingFollowerCount,
                                        scalerEngagingFollowingCount])
        '''
        ## first vectorize for spark... meh
        assembler = VectorAssembler(inputCols=["tweet_timestamp", "engaged_with_user_account_creation",
                                   "engaging_user_account_creation", "engaged_with_user_follower_count",
                                  "engaged_with_user_following_count", "engaging_user_follower_count",
                                  "engaging_user_following_count"], outputCol="numeric_features")
        

        numericScaler = MinMaxScaler(inputCol="numeric_features", outputCol="numeric_scaled")
        scalePipeline = Pipeline(stages=[assembler, numericScaler])
        self.scaleModel = scalePipeline.fit(outputDF)
        outputDF = self.scaleModel.transform(outputDF)
        
        # drop numeric columns
        outputDF = outputDF.drop(*["tweet_timestamp", "engaged_with_user_account_creation",
                                   "engaging_user_account_creation", "engaged_with_user_follower_count",
                                  "engaged_with_user_following_count", "engaging_user_follower_count",
                                  "engaging_user_following_count", "numeric_features"])
        
        outputDF = self._assemble(outputDF)
        self.processedTrainDF = outputDF
        
        # might not need
        # transform boolean to 0-1 column... first one has to change the type in the schema though 
        #data = data.select("engaging_user_is_verified", "engaged_with_user_is_verified", "engaged_follows_engaging")\
        #    .replace(["false","true"], ["0","1"]).show()
        
        
    '''
        Preprocess test file if given...
    '''
    def _preprocessTest(self):
        test = self.testFile
        
        ### repeat all the steps that went place for train
        # Drop unnecessary cols
        ### drop unused ids for classification
        test = test.drop("engaged_with_user_id").drop("engaged_user_id")\
                    .drop("present_links").drop("present_domains")
        
        # Split the text tokens to valid format
        textTokenizer = RegexTokenizer(inputCol="text_tokens",outputCol="vector", pattern="\t")
        test = textTokenizer.transform(test)
        hashtagTokenizer = RegexTokenizer(inputCol="hashtags",outputCol="hashtag_tokens", pattern="\t")
        test = hashtagTokenizer.transform(test.fillna("none", subset=["hashtags"]))
        
        test = test.drop("text_tokens").withColumnRenamed("vector", "text_tokens")
        test = test.drop("hashtags").withColumnRenamed("hashtag_tokens", "hashtags")
        
        regexTokenizer = RegexTokenizer(inputCol="present_media", outputCol="media_list")
        test = regexTokenizer.transform(test.fillna("none", subset=["present_media"]))
        test = test.drop("present_media").withColumnRenamed("media_list", "present_media")
        test = test.withColumn("present_media2", test["present_media"].cast(StringType()))
        test = test.drop("present_media").withColumnRenamed("present_media2", "present_media")
        
        ### REUSE MODELS FROM PROCESSING TRAIN
        test = self.indexerModel.transform(test)
        test = self.encoderModel.transform(test)
        test = test.drop(*["tweet_type","tweet_type_id", "language","language_id","present_media","present_media_id"])
        # tf/idf text + hashtags
        test = self.tfidfModel.transform(test)
        test = test.drop(*["hashtags", "hashtagsTF", "text_tokens", "tweet_text_TF"])

        # scale numeric
        test = self.scaleModel.transform(test)
        
        # drop numeric columns
        test = test.drop(*["tweet_timestamp", "engaged_with_user_account_creation",
                                   "engaging_user_account_creation", "engaged_with_user_follower_count",
                                  "engaged_with_user_following_count", "engaging_user_follower_count",
                                  "engaging_user_following_count", "numeric_features"])
        
        # rename target columns
        test = test.withColumnRenamed("like_timestamp", "like")#.drop("like_timestamp")
        test = test.withColumnRenamed("retweet_timestamp", "retweet")
        test = test.withColumnRenamed("reply_timestamp", "reply")
        test = test.withColumnRenamed("retweet_with_comment_timestamp", "retweet_comment")
        test = self._assemble(test)
        
        #outputDF = outputDF.withColumn("like", when(outputDF["like_timestamp"].isNull(), 0).otherwise(1))
        #outputDF = outputDF.withColumn("retweet", when(outputDF["retweet_timestamp"].isNull(), 0).otherwise(1))
        #outputDF = outputDF.withColumn("reply", when(outputDF["reply_timestamp"].isNull(), 0).otherwise(1))
        #outputDF = outputDF.withColumn("retweet_comment", when(outputDF["retweet_with_comment_timestamp"].isNull(), 0).otherwise(1))

        
        self.processedTestDF = test
    
    '''
        returns small dataframe that explains the values of the oneHotEncoder step, this might be needed
        for mapping the encodings back to the original values
    '''
    def explainOneHot(self):
        return self.explainOneHotDF
