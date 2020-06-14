
import pandas as pd

outpath = "../output/"
#preds = ["test"]
preds = ["like", "retweet", "reply", "retweet_comment"]
cols = ["tweet_id", "engaging_user_id", "prediction"]

for out in preds:
    fpath = outpath+out+"_out.csv"
    df = pd.read_csv(fpath)
    print("finished reading "+out)
    df[cols].to_csv(outpath+out+"_format.csv", header=False, index=False)
    print("finished writing " + out)
