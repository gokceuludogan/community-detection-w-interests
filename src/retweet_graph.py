import re
import pandas as pd
import rdflib
file1 = open('unique_retweets.txt', 'r')
Lines = file1.readlines()

regex = re.compile(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]')
pattern = re.compile(r'\s+')

count = 0
uniques_retweets=[]
# Strips the newline character
for line in Lines:
    count += 1
    if len(line.strip())>0:
        uniques_retweets.append(line.strip())


print(len(uniques_retweets))


df = pd.read_csv("wd_annotated_tweets.csv", sep=",")
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

        x = retweets.split(";")
        for k in range(len(x)):
                    if x[k].find(",")>=0:
                        x[k] = x[k][0:x[k].rindex(",")]
                    x[k] = x[k].lower()
                    x[k] = re.sub(pattern, '', x[k])
                    x[k] = re.sub(regex, '', x[k])
                    if x[k] in uniques_retweets:
                        f[uniques_retweets.index(x[k])] += 1


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
    rtwt=df_res.loc[inx]

    inds=[]
    weights=[]
    for ii, e in enumerate(rtwt):
        if e!=0:
            inds.append(ii)
            weights.append(e)

    rts=[]
    for k in inds:
        rts.append(uniques_retweets[k])

    inxx=0
    for j in rts:
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

