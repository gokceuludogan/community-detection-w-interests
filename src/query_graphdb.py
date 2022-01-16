import pandas as pd
import time
from SPARQLWrapper import SPARQLWrapper, JSON


sparql = SPARQLWrapper("http://localhost:7200/repositories/CommunityDB")
ONTOLOGY_NAME = 'twitter-interests'
ONTOLOGY_URI = 'http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#'

def query(q):
	sparql.setQuery(q)
	sparql.setReturnFormat(JSON)
	time_start = time.time()
	results = sparql.query().convert()
	print("--- %s seconds ---" % (time.time() - time_start))
	return pd.json_normalize(results['results']['bindings'])


def get_interests():
	q = f"""
	PREFIX {ONTOLOGY_NAME}: <{ONTOLOGY_URI}>
	prefix dbo: <http://dbpedia.org/ontology/>

	SELECT DISTINCT ?user ?interest ?interestType
	WHERE {{
	      ?user {ONTOLOGY_NAME}:hasInterest ?interest .
	      ?interest a ?interestType . 
	}}
	"""
	return query(q)

def get_interactions(interaction_type='mention'):
	q = f"""
	PREFIX {ONTOLOGY_NAME}: <{ONTOLOGY_URI}>
	prefix dbo: <http://dbpedia.org/ontology/>

	SELECT DISTINCT ?fromUser ?toUser ?weight
	    WHERE {{
	      ?interaction {ONTOLOGY_NAME}:fromUser ?fromUser .
	      ?interaction {ONTOLOGY_NAME}:toUser ?toUser .
	      ?interaction {ONTOLOGY_NAME}:scale ?weight .
	      ?interaction {ONTOLOGY_NAME}:hasType "{interaction_type}" .
	    }}
	"""
	return query(q)

# get_interactions()
# get_interests()
# print(df.columns)
# print(df)
# print(df.iloc[0])


