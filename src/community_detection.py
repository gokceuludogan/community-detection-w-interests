import networkx as nx
import json
import time
import logging
from pathlib import Path
from cdlib import algorithms, evaluation
from ontology import Ontology


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', level=logging.INFO)

logging.info('Loading ontology...')
ontology = Ontology('../data/populated_interests.owl') # sample.owl')

logging.info('Retrieving mentions...')
mentions = ontology.get_all_interactions('mention')

logging.info('Retrieving retweets...')
retweets = ontology.get_all_interactions('retweets')

mention_dict = {(user1, user2): weight for user1, user2, weight in mentions}
retweet_dict = {(user1, user2): weight for user1, user2, weight in retweets}

logging.info('Retrieving interests...')

interests = ontology.get_class_based_interests()
#interests = ontology.get_all_interests()

logging.info('Finding user interest similarities')
interest_dict = {}
for user1, interest1, user2, interest2, interest_type in interests:
	if (user1, user2) not in interest_dict:
		interest_dict[(user1, user2)] = {interest_type: {'common': interest1 == interest2, 'count': 1}}
	else:
		if interest_type not in interest_dict[(user1, user2)]:
			interest_dict[(user1, user2)][interest_type] = {'common': interest1 == interest2, 'count': 1}
		else:
			interest_dict[(user1, user2)][interest_type]['common'] += (interest1 == interest2)
			interest_dict[(user1, user2)][interest_type]['count'] += 1

interest_user_sim = {}
for user1, user2 in interest_dict:
	counts = interest_dict[(user1, user2)]
	interest_user_sim[(user1, user2)] = [counts[interest_type]['common'] / counts[interest_type]['count'] for interest_type in counts]	
	


logging.info('Constructing homogenoues graph')
user_interest_dict = {pair: sum(class_sim)/len(class_sim) for pair, class_sim in interest_user_sim.items()}

all_pairs = set(mention_dict).union(set(retweet_dict)).union(set(user_interest_dict))
edges = {key: mention_dict.get(key, 0) + retweet_dict.get(key, 0) + user_interest_dict.get(key, 0) for key in all_pairs}

nodes = set([node for pair in edges.keys() for node in pair])

G = nx.Graph()

node_mapping_id = {}
node_mapping = {}
for ix, node in enumerate(nodes):
	G.add_node(ix)
	node_mapping_id[ix] = node
	node_mapping[node] = ix

for pair, weight in edges.items():
	G.add_edge(node_mapping[pair[0]], node_mapping[pair[1]], weight=weight)

logging.info('Running Leiden algorithm...')
coms = algorithms.leiden(G)
communities = coms.communities

logging.info('Evaluation...')
avg_internal_degree = coms.average_internal_degree(summary=False)
link_modularity = coms.link_modularity() # G, communities)
tpr = coms.triangle_participation_ratio() # G, communities)
z_modularity = coms.z_modularity() # G, communities)

# avg_distance = evaluation.avg_distance(G, communities)
# surprise = evaluation.surprise(G, communities)
# purity = evaluation.purity(communities)

path = Path('../results')
path.mkdir(exist_ok=True, parents=True)

with open(path / 'leiden_predictions.json', 'w') as f:
	f.write(json.dumps({'node_id': node_mapping, 
		'id_node': node_mapping_id, 
		'communities': communities
		}))

with open(path / 'leiden_scores.json', 'w') as f:
	f.write(json.dumps({'avg_internal_degree': avg_internal_degree,
		#'avg_distance': avg_distance, 
		'z_modularity': z_modularity, 'link_modularity': link_modularity, 
		#'surprise': surprise, 
		'tpr': tpr#, 'purity': purity
		}))
