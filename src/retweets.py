import rdflib
import time
import pandas as pd
import numpy as np
import re, string

regex = re.compile(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~🔥]')
pattern = re.compile(r'\s+')

def uniques(retweets):
    uniques_retweets = []

    for j in range(len(retweets)):
        if str(retweets[j])!="nan":
            rts = rtwts[j].split(";")
            for k in range(len(rts)):
                if rts[k].find(",")>=0:
                    rts[k] = rts[k][0:rts[k].rindex(",")]
                rts[k] = rts[k].lower()
                rts[k] = re.sub(pattern, '', rts[k])
                rts[k] = re.sub(regex, '', rts[k])
                if rts[k] not in uniques_retweets:
                    uniques_retweets.append(rts[k])

    return uniques_retweets



df = pd.read_csv("wd_annotated_tweets.csv", sep=",")
rtwts = df["retweet"].values
users = df["UserName"].values
uniques_retweets = uniques(rtwts)


textfile = open("unique_retweets.txt", "w")
for element in uniques_retweets:
    textfile.write(element + "\n")
textfile.close()

