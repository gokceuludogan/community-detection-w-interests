import re
import pandas as pd
import rdflib
file1 = open('uniques_mentions_1000.txt', 'r')
Lines = file1.readlines()

regex = re.compile(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]')
pattern = re.compile(r'\s+')

count = 0
uniques_mentions=[]
# Strips the newline character
for line in Lines:
    count += 1
    if len(line.strip())>0:
        uniques_mentions.append(line.strip())


print(len(uniques_mentions))


df = pd.read_csv("D:\\cmpe_58H\\wd_annotated_tweets.csv", sep=",")
#print(df.iloc[40])
cnt=0
usr=[]
weights=[]
for index, row in df.iterrows():

    username = row["UserName"]
    mentions = row["mentions"]
    if str(mentions)!="nan" and str(username)!="nan":
        username = username.lower()
        username = re.sub(pattern, '', username)
        username = re.sub(regex, '', username)
        f = [0] * 1000
        cnt=cnt+1

        ms = mentions.split(";")
        for k in range(len(ms)):
                    if ms[k].find(",")>=0:
                        ms[k] = ms[k][0:ms[k].rindex(",")]
                    ms[k] = ms[k].lower()
                    ms[k] = re.sub(pattern, '', ms[k])
                    ms[k] = re.sub(regex, '', ms[k])
                    if ms[k] in uniques_mentions:
                        f[uniques_mentions.index(ms[k])] += 1


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
    men=df_res.loc[inx]


    print(i)


    inds=[]
    weights=[]
    for ii, e in enumerate(men):
        if e!=0:
            inds.append(ii)
            weights.append(e)

    mntns=[]
    for k in inds:
        mntns.append(uniques_mentions[k])

    inxx=0
    for j in mntns:
        g.add((node, rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#interacts"),
               rdflib.URIRef("http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionLevel/"+str(user)+"_"+str(j)+"_mention")))

        g.add((rdflib.URIRef("http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionLevel/" + str(user) + "_" + str(j) + "_mention"), rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#mention"),
               rdflib.URIRef("http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#User/" + str(j))))

        g.add((rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionLevel/" + str(
                user) + "_" + str(j) + "_mention"), rdflib.URIRef(
            "http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#InteractionScale"),
               rdflib.URIRef(str(weights[inxx]))))

        inxx =inxx+1

    #print(user)
    #print(mntns)

twitter_mentions_fh = open("D:\\cmpe_58H\\twitter_mentions_v2.rdf", "w",  encoding="utf-8")
twitter_mentions_fh.write(g.serialize())
twitter_mentions_fh.close()

print(g.serialize())