import networkx as nx
import json
import time
import logging
from pathlib import Path
from cdlib import algorithms, evaluation

import json
import time
import logging
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from query_graphdb import *

	
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

logging.info('Loading data...')

logging.info('Retrieving mentions...')
mentions = get_interactions('mention')

logging.info('Retrieving retweets...')
retweets = get_interactions('retweet')

logging.info('Retrieving interests...')

interests = get_interests()

mention_dict = mentions[['fromUser.value', 'toUser.value', 'weight.value']].set_index(['fromUser.value','toUser.value']).to_dict()['weight.value']
retweet_dict = retweets[['fromUser.value', 'toUser.value', 'weight.value']].set_index(['fromUser.value','toUser.value']).to_dict()['weight.value'] 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', level=logging.INFO)

size = 50000
interest_level = 'superclass'
algorithm = 'rb_pots'
df_interests = pd.read_csv(f'E:\\Asus-zenbook-d\\boun-phd\\fall21\\CMPE58H\\project\\ssw-project\\data\\vectors_{size}\\{interest_level}\\interest_similarity.csv')
interest_dict = df_interests.groupby(['user1', 'user2'])['score'].mean().reset_index().set_index(['user1','user2']).to_dict()['score']

all_pairs = set(mention_dict).union(set(retweet_dict)).union(set(interest_dict))
key = list(all_pairs)[0]

from itertools import product
alphas = [-1, 0, 1]
for alpha1, alpha2, alpha3 in list(product(alphas, repeat=3))[::-1]:
	if alpha1 == 0 and alpha2 == 0 and alpha3 == 0:
		continue
	path = Path(f'../results_{size}/{algorithm}/{interest_level}/weights_{alpha1}_{alpha2}_{alpha3}')
	if path.is_dir():
		continue
	n1, n2, n3 = alpha1 * -1, alpha2 * -1, alpha3 * -1
	if Path(f'../results_{size}/{algorithm}/{interest_level}/weights_{n1}_{n2}_{n3}').is_dir():
		continue
	logging.info(f'Weights: {alpha1} {alpha2} {alpha3}')
	# alpha1, alpha2, alpha3 = 0, 1, 0

	edges = {key:  (alpha1 * float(mention_dict.get(key, 0)) + alpha2 * float(retweet_dict.get(key, 0)) + alpha3 *  float(interest_dict.get(key, 0))) for key in all_pairs}
	edges = {key: value for key, value in edges.items() if value != 0}

	logging.info(f'Number of edges {len(edges)}')
	nodes = set([node for pair in edges.keys() for node in pair])
	logging.info(f'Number of nodes {len(nodes)}')

	G = nx.Graph()

	node_mapping_id = {}
	node_mapping = {}
	for ix, node in enumerate(nodes):
		G.add_node(ix)
		node_mapping_id[ix] = node
		node_mapping[node] = ix

	for pair, weight in edges.items():
		G.add_edge(node_mapping[pair[0]], node_mapping[pair[1]], weight=weight)

	logging.info('Running algorithm...')
	# coms = algorithms.surprise_communities(G)
	coms = algorithms.rb_pots(G)

	communities = coms.communities
	logging.info(f'Number of communities: {len(communities)}')
	logging.info(f'Community Sizes top 100 {",".join([str(len(community)) for community in communities][:100])}')
	logging.info('Evaluation...')
	logging.info('Calculating average_internal_degree')
	avg_internal_degree = coms.average_internal_degree(summary=True).score
	logging.info('Calculating conductance')
	conductance = coms.conductance(summary=True).score
	# logging.info('Calculating triangle_participation_ratio')
	# tpr = coms.triangle_participation_ratio(summary=True).score # G, communities)
	# logging.info('Calculating z_modularity')
	# z_modularity = coms.z_modularity().score # G, communities)
	logging.info('Calculating newman_girvan_modularity')
	ng = coms.newman_girvan_modularity().score 
	# logging.info('Calculating avg_distance')
	# avg_distance = coms.avg_distance().score
	logging.info('Calculating surprise')
	surprise = coms.surprise().score


	path.mkdir(exist_ok=True, parents=True)

	with open(path / 'predictions.json', 'w') as f:
		f.write(json.dumps({'node_id': node_mapping, 
			'id_node': node_mapping_id, 
			'communities': communities
			}))

	with open(path / 'scores.json', 'w') as f:
		f.write(json.dumps({'avg_internal_degree': avg_internal_degree,
			#'avg_distance': avg_distance, 
			#'z_modularity': z_modularity, 
			'surprise': surprise, 
			'conductance': conductance,
			#'tpr': tpr,
			'newman_girvan': ng
			}))
