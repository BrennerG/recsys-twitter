
import pandas as pd
from os import listdir
from os.path import isfile, join
outpath = "../output/rf/"
files = [f for f in listdir(outpath) if isfile(join(outpath, f))]
for i in range(len(files)):
    files[i] = files[i].split(".")[0]

cols = ["tweet_id", "engaging_user_id", "positive_probability"]

for out in files:
    fpath = outpath+out+".csv"
    df = pd.read_csv(fpath)
    print("finished reading "+out)
    df["positive_probability"] = df.apply(lambda row: row["probability"][1:-1].split(",")[1], axis=1)
    df[cols].to_csv(outpath+out+"_format.csv", header=True, index=False)
    print("finished writing " + out)
