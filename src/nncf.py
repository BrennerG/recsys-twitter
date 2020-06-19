import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import twitter_preproc
import importlib
importlib.reload(twitter_preproc)
from twitter_preproc import *
from operator import attrgetter
import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from twitter_preproc import twitter_preproc

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.optim as optim

import sys, traceback

# Building Spark Context
conf = SparkConf().setAll([('spark.executor.memory', '5g'), ('spark.executor.cores', '30'), ('spark.driver.memory','8g')])
spark = SparkSession  .builder   .appName("Pipelinee")   .config(conf=conf)   .getOrCreate()
sc = spark.sparkContext

# Small datasets
# train = "///tmp/traintweet_1000.tsv"
# test = "///user/e11920598/test_1000.tsv"

# Full datasets
# the full train file has 121.386.431 lines
# the full test file has 12.434.838 lines
train = "///user/pknees/RSC20/training.tsv"
# test= "///user/pknees/RSC20/test.tsv"

preproc = twitter_preproc(spark, sc, train)
traindata = preproc.getDF()

# Create Indices & One-Hot Vectors
stages = []
stages.append(StringIndexer(inputCol="tweet_id", outputCol="tweet_id_indexed"))
stages.append(OneHotEncoder(inputCol="tweet_id_indexed",  outputCol="tweet_id_ohe"))
stages.append(StringIndexer(inputCol="engaging_user_id", outputCol="user_id_indexed"))
stages.append(OneHotEncoder(inputCol="user_id_indexed",  outputCol="user_id_ohe"))
pipeline = Pipeline(stages=stages)

model = pipeline.fit(traindata)
traindata_ohe = model.transform(traindata).select(["tweet_id_ohe","user_id_ohe","like"])

df = pd.DataFrame(traindata_ohe.take(1000), columns=traindata_ohe.columns)

# map function to convert from spark vectors to sparse numpy csr matrix
def as_matrix(vec):
    data, indices = vec.values, vec.indices
    shape = 1, vec.size
    return csr_matrix((data, indices, np.array([0, vec.values.size])), shape)

# calls as_matrix
def get_pytorch_sparse(attr):
    features = traindata_ohe.rdd.map(attrgetter(attr))
    mats = features.map(as_matrix)
    mat = mats.reduce(lambda x, y: vstack([x, y]))
    return mat


# Neural Network time :)
# convert to pytorch format
def transform_to_sparse_tensor(data):
    coo = data.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


# convert matrices to pytorch tensor format
tweet_sparse = get_pytorch_sparse("tweet_id_ohe")
user_sparse = get_pytorch_sparse("user_id_ohe")
tweet_sparse = transform_to_sparse_tensor(tweet_sparse).to_dense()
user_sparse = transform_to_sparse_tensor(user_sparse).to_dense()

# create target variables in correct format
y = torch.FloatTensor(traindata_ohe.select("like").collect()) 
target = y  
target = target.view(1, -1).t()


# Define Neural Network
class Net(nn.Module):

    def __init__(self, users, items, k):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(users.shape[1], k)
        self.dense2 = nn.Linear(items.shape[1], k)
        self.fc1 = nn.Linear(2*k, k)
        self.fc2 = nn.Linear(k, math.floor(k/2))
        self.fc3 = nn.Linear(math.floor(k/2), 1)

    def forward(self, users, items):
        users = F.relu(self.dense1(users))
        items = F.relu(self.dense2(items))
        # concat users and items into 1 vector
        x = torch.cat((users, items), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        sigfried = nn.Sigmoid()
        x = sigfried(self.fc3(x))
        return x


# Initialize Neural Network
net = Net(user_sparse, tweet_sparse, 100)
output = net(user_sparse, tweet_sparse)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
print("\n\n", net)

print("\nStart Training")
# Specifiy Training Loop
n_epochs = 1
batch_size = 100000

for epoch in range(n_epochs):
    print("epoch ", epoch)

    try:
    # X is a torch Variable
        permutation = torch.randperm(user_sparse.size()[0])

        for i in range(0,user_sparse.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x_user = user_sparse[indices]
            batch_x_tweet = tweet_sparse[indices]
            batch_y = target[indices]

            # in case you wanted a semi-full example
            outputs = net.forward(batch_x_user, batch_x_tweet)
            loss = criterion(outputs,batch_y)

            loss.backward()
            optimizer.step()
            print(loss)

    except:
        traceback.print_stack()

# Save the model
PATH = './NNCF_model_save.pth'
torch.save(net.state_dict(), PATH)


print("\n\nDONE. model saved to ", PATH, "\n\n")
