from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import importlib
import twitter_preproc
import nnpreprocessor
importlib.reload(nnpreprocessor)
from NNCFNet import Net
import torch
import torch.nn as nn
import torch.optim as optim
import sys, traceback
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np


# Building Spark Context
conf = SparkConf().setAll([('spark.executor.memory', '32g'), ('spark.executor.instances','8'),('spark.executor.cores', '12'), ('spark.driver.memory','64g'), ('spark.driver.memoryOverhead', '64g')])
spark = SparkSession.builder.appName("nncf_submission").config(conf=conf).getOrCreate()
sc = spark.sparkContext

# Get Data & Basic preprocessing
train = "///user/pknees/RSC20/training.tsv"
preproc = twitter_preproc.twitter_preproc(spark, sc, train, MF=True)
traindata = preproc.getDF()

# NN specific preprocessing
nnp = nnpreprocessor.NNPreprocessor()
engagement = 'retweet_comment'
tweets, users, target = nnp.nn_preprocess(traindata)

# Initalize Hyperparameters
k = 32
n_epochs = 2
batch_size = 1024

# Initialize Neural Network
net = Net(users.shape[1], tweets.shape[1], k)
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.BCELoss()
output = net(users, tweets)

# Start training
for epoch in range(n_epochs):

    permutation = torch.randperm(users.size()[0])

    for i in range(0,users.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x_user = users[indices]
        batch_x_tweet = tweets[indices]
        batch_y = target[indices]

        outputs = net.forward(batch_x_user, batch_x_tweet)
        loss = criterion(outputs,batch_y)
        loss.backward()
        optimizer.step()

# Save neural network for evaluation
PATH = '../misc/NNCF_model_save.pth'
torch.save(net.state_dict(), PATH)
print("\n\nDONE. model saved to ", PATH)