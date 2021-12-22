import rdflib
import time
import pandas as pd
import numpy as np
import re, string

regex = re.compile(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]')
pattern = re.compile(r'\s+')

def uniques(retweets):
    unique_retweets = []
    for j in range(len(retweets)):
        if str(retweets[j])!="nan":
            rt = rts[j]
            #rt = rt.lower()
            #rt = re.sub(pattern, '', rt)
            #rt = re.sub(regex, '', rt)
            if (rt not in unique_retweets):
                unique_retweets.append(rt)
    return unique_retweets



df = pd.read_csv("wd_annotated_tweets.csv")
print(len(df))
rts = df["retweet"].values
users = df["UserName"].values
unique_retweets = uniques(rts)


textfile = open("unique_retweets.txt", "w")
for element in unique_retweets:
    textfile.write(element + "\n")
textfile.close()

