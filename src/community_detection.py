import networkx as nx
import json

from pathlib import Path
from cdlib import algorithms, evaluation
from ontology import Ontology

ontology = Ontology('../data/populated_interests.owl')
mentions = ontology.get_all_interactions('mention')
retweets = ontology.get_all_interactions('retweets')

mention_dict = {(user1, user2): weight for user1, user2, weight in mentions}
retweet_dict = {(user1, user2): weight for user1, user2, weight in retweets}

edges = {key: mention_dict.get(key, 0) + retweet_dict.get(key, 0) for key in set(mention_dict.keys()).union(set(retweet_dict.keys()))}

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

coms = algorithms.leiden(G)
communities = coms.communities

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



	