import re
import pandas as pd
import rdflib
file1 = open('unique_retweets.txt', 'r')
Lines = file1.readlines()

regex = re.compile(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]')
pattern = re.compile(r'\s+')

count = 0
unique_retweets=[]
# Strips the newline character
for line in Lines:
    count += 1
    if len(line.strip())>0:
        unique_retweets.append(line.strip())



df = pd.read_csv("wd_annotated_tweets.csv", sep=",")
#print(df.iloc[40])
cnt=0
usr=[]
weights=[]
for index, row in df.iterrows():

    username = row["UserName"]
    retweets = row["retweet"]
    if str(retweets)!="nan" and str(username)!="nan":
        username = username.lower()
        username = re.sub(pattern, '', username)
        username = re.sub(regex, '', username)
        f = [0] * 1000
        cnt=cnt+1

        rts = retweets.split(";")
        for k in range(len(rts)):
                    if rts[k].find(",")>=0:
                        rts[k] = rts[k][0:rts[k].rindex(",")]
                    rts[k] = rts[k].lower()
                    rts[k] = re.sub(pattern, '', rts[k])
                    rts[k] = re.sub(regex, '', rts[k])
                    if rts[k] in unique_retweets:
                        f[unique_retweets.index(rts[k])] += 1


        usr.append(username)
        weights.append(f)

        if index >= 1800:
            break


print(cnt)
df_res = pd.DataFrame(weights,columns=range(1000))
df_res = (df_res - df_res.mean()) / (df_res.max() - df_res.min())
df_res.fillna(0)
print(df_res.shape)

g = rdflib.Graph()
for i, inx in enumerate(df_res.index):
    user=usr[i]
    user = user.lower()
    user = re.sub(pattern, '', user)
    user = re.sub(regex, '', user)
    tweet_uri = "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#User/" + str(user)
    node = rdflib.URIRef(tweet_uri)
    rt=df_res.loc[inx]




    inds=[]
    weights=[]
    for ii, e in enumerate(rt):
        if e!=0:
            inds.append(ii)
            weights.append(e)

    rtwts=[]
    for k in inds:
        rtwts.append(unique_retweets[k])

    inxx=0
    for j in rtwts:
        g.add((node, rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#interacts"),
               rdflib.URIRef("http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionLevel/"+str(user)+"_"+str(j)+"_retweet")))

        g.add((rdflib.URIRef("http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionLevel/" + str(user) + "_" + str(j) + "_retweet"), rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#retweet"),
               rdflib.URIRef("http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#User/" + str(j))))

        g.add((rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionLevel/" + str(
                user) + "_" + str(j) + "_retweet"), rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionScale"),
               rdflib.URIRef(str(weights[inxx]))))

        inxx =inxx+1



twitter_retweets_fh = open("twitter_retweets.rdf", "w",  encoding="utf-8")
twitter_retweets_fh.write(g.serialize())
twitter_retweets_fh.close()

