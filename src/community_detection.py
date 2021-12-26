import networkx as nx
import json
import time
import logging
from pathlib import Path
from cdlib import algorithms, evaluation
from ontology import Ontology


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', level=logging.INFO)

logging.info('Loading data...')

with open('../data/user_similarities.json') as f:
	data = json.loads(f.read())

edges = {key: value['mention'] + value['retweet'] + value['interest'] for key, value in data.items()}
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
