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
sklearn
numpy
streamlit
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
  
* Execute [DataProcessing notebook](notebooks/DataProcessing.ipynb) to merge tweets 

#### Annotate Tweets with DBpedia

```bash
python tweet_processor.py --input tweets.csv --output db_annotated_tweets.csv
```

#### Prepare Data

Execute [Triplets notebook](notebooks/Triplets.ipynb) to prepare data for populating ontology

### Populate Ontology

```bash
python construct_graph.py
```

### Detect Communities

#### Prepare weights

```bash
python prepare_input_graph.py
```
#### Run algorithm
```bash
python community_detection.py
```

### Results
* Once communities are detected,
  * Export csv files for importing Gephi.

  ```bash
  python export_csv.py
  ```

  * Run application showing the most important words for each community. 
  ```bash
  streamlit run app.py
  ```
