import rdflib
import time
import pandas as pd
import numpy as np
import re, string

regex = re.compile(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]')
pattern = re.compile(r'\s+')

def uniques(mentions):
    uniques_mentions = []

    for j in range(len(mentions)):
        print(j)
        if str(mentions[j])!="nan":
            ms = ments[j].split(";")
            for k in range(len(ms)):
                ms[k] = ms[k][0:ms[k].rindex(",")]
                ms[k] = ms[k].lower()
                ms[k] = re.sub(pattern, '', ms[k])
                ms[k] = re.sub(regex, '', ms[k])
                if ms[k] not in uniques_mentions:
                    uniques_mentions.append(ms[k])

        #the mentions taken can be controlled with below statement
        if len(uniques_mentions)>=1000:
            break


    return uniques_mentions



df = pd.read_csv("D:\\cmpe_58H\\wd_annotated_tweets.csv", sep=",")
print(len(df))
ments = df["mentions"].values
users = df["UserName"].values
uniques_mentions = uniques(ments)


textfile = open("uniques_mentions_1000.txt", "w")
for element in uniques_mentions:
    textfile.write(element + "\n")
textfile.close()

