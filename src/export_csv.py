import pandas as pd
import json
from pathlib import Path
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

size = 25000
algorithm = 'rb_pots'
interest_level = 'subclass'
df_interests = pd.read_csv(f'../data/vectors_{size}/{interest_level}/interest_similarity.csv')
interest_dict = df_interests.groupby(['user1', 'user2'])['score'].mean().reset_index().set_index(['user1','user2']).to_dict()['score']

all_pairs = set(mention_dict).union(set(retweet_dict)).union(set(interest_dict))
key = list(all_pairs)[0]


alpha1 = 1
alpha2 = 1
alpha3 = 1
path = Path(f'../results_{size}/{algorithm}/{interest_level}/weights_{alpha1}_{alpha2}_{alpha3}')
data = json.loads(open(path / 'predictions.json').read())
id_to_node = data['id_node']

edges = {key:  (alpha1 * float(mention_dict.get(key, 0)) + alpha2 * float(retweet_dict.get(key, 0)) + alpha3 *  float(interest_dict.get(key, 0))) for key in all_pairs}
edges = {key: value for key, value in edges.items() if value != 0}

nodes = []
filtered_edges = []
for cid, community in enumerate(data['communities']):
	random.shuffle(community)
	for user in community:
		number = random.randint(1, 5)
		if number == 1:
			label = id_to_node[str(user)].split('#')[-1]
			nodes.append((label, label, cid))
node_labels = [i[0] for i in nodes]
filtered_nodes = []

for pair, value in tqdm(edges.items()):
	label0 = pair[0].split('#')[-1]
	label1 = pair[1].split('#')[-1]
	if label0 in node_labels and label1 in node_labels:
		filtered_edges.append((label0, label1, value))
		filtered_nodes.append(nodes[node_labels.index(label0)])
		filtered_nodes.append(nodes[node_labels.index(label1)])

pd.DataFrame(filtered_nodes, columns=['Id', 'Label', 'Community']).to_csv(f'nodes_{alpha1}_{alpha2}_{alpha3}.csv', index=False)
pd.DataFrame(filtered_edges, columns=['Source', 'Target', 'weight']).to_csv(f'edges_{alpha1}_{alpha2}_{alpha3}.csv', index=False)
