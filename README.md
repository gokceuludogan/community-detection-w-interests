# Twitter Community Detection Leveraging User Interests


## Requirements

```
spacyopentapioca
tweet-preprocessor
spacy-dbpedia-spotlight
spacy
pandas
rdflib
Scweet
tweepy
cdlib
```

Optional: `pydotplus` for visualizing ontology. 


## Pipeline

### Data Collection 

#### Collect tweets containing keywords 

* With `Tweepy`

  ```bash
  python --method api --data keyword
  ```
  
* With `Scweet`

  ```bash 
  python --method scaper --data keyword
  ```
  
* Execute [DataProcessing notebook](notebooks/DataProcessing.ipynb) to merge tweets and extract users at least tweet three times containing keywords. 

#### Collect latest tweets of users

```bash
python --method tweepy --data user
```

#### Annotate Tweets with DBpedia & Wikidata 

```bash
python tweet_processor.py --input tweets.csv --output annotated_tweets.csv
```

#### Prepare Data

Execute [Triplets notebook](notebooks/Triplets.ipynb) to prepare data for populating ontology

### Populate Ontology

```bash
python construct_graph.py
```

### Detect Communities
```bash
python community_detection.py
```
