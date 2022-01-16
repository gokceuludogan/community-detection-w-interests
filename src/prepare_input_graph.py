import json
import time
import logging
import numpy as np
import torch
import random
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from ontology import Ontology
from tqdm import tqdm
from query_graphdb import *

def pw_cosine_distance(input_a, input_b):
   normalized_input_a = torch.nn.functional.normalize(input_a)  
   normalized_input_b = torch.nn.functional.normalize(input_b)
   res = torch.mm(normalized_input_a, normalized_input_b.T)
   return res
	
logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')



logging.info('Retrieving mentions...')
mentions = get_interactions('mention')

logging.info('Retrieving retweets...')
retweets = get_interactions('retweet')

logging.info('Retrieving interests...')

interests = get_interests()

mention_dict = mentions[['fromUser.value', 'toUser.value', 'weight.value']].set_index(['fromUser.value','toUser.value'])['weight.value'].to_dict() 
retweet_dict = retweets[['fromUser.value', 'toUser.value', 'weight.value']].set_index(['fromUser.value','toUser.value'])['weight.value'].to_dict() 
interest_freqs = interests.groupby(['interest.value', 'interestType.value']).size().reset_index(name='counts')
interest_triples = interests[['user.value', 'interest.value', 'interestType.value']].to_records()
interest_freqs = interests.groupby(['interest.value', 'interestType.value']).size().reset_index(name='counts')
interest_freqs['counts'] = interest_freqs['counts'].apply(lambda x: 1/x)
interest_weights = interest_freqs[interest_freqs['counts'] < 0.5].set_index('interest.value').to_dict()['counts']


all_users = set(mentions['fromUser.value'].tolist() + mentions['toUser.value'].tolist() + \
 			retweets['fromUser.value'].tolist() + retweets['toUser.value'].tolist() + interests['user.value'].tolist())

num_of_users = 25000
sampled_users = random.sample(all_users, num_of_users)
base = 'http://dbpedia.org/ontology/'
subclasses = ['Person', 'Group', 'Organization', 'TopicalConcept']
superclasses = ['Agent', 'TopicalConcept']
classes = {'subclass': subclasses, 'superclass': superclasses}

vector_path = Path(f'../data/vectors_{num_of_users}')
for name, class_ in classes.items(): 
	interests_dict = {f'{base}{c}': {} for c in class_}
	interests_ix_dict = {f'{base}{c}': {} for c in class_}
	user_interests = {}
	for ix, user, interest, interest_type in interest_triples:
		if interest_type in interests_dict and user in sampled_users and interest in interest_weights: 
			if interest in interests_dict[interest_type]:
				index = interests_dict[interest_type][interest]
			else:
				index = len(interests_dict[interest_type]) 
				interests_dict[interest_type][interest] = index
				interests_ix_dict[interest_type][index] = interest

			if user in user_interests:
				if interest_type in user_interests[user]:
					user_interests[user][interest_type].add(index)
				else:
					user_interests[user][interest_type] = {index}
			else:
				user_interests[user] = {interest_type: {index}}

	interest_lengths = {i_type: len(individuals) for i_type, individuals in interests_dict.items()}
	user_vectors = {}
	interest_vectors = {i_type: [] for i_type in interests_dict.keys()}
	user_indices = {user: ix for ix, user in enumerate(user_interests.keys())}
	user_indices_to_id = {ix: user for user, ix in user_indices.items()}
	logging.info('Vectorizing class based interests...')
	for user, values in user_interests.items():
		user_vectors[user] = {}
		for i_type, individuals in values.items():
			vector = [interest_weights.get(interests_ix_dict[i_type][i], 0) if i in individuals else 0 for i in range(interest_lengths[i_type])]
			user_vectors[user][i_type] = vector 
			interest_vectors[i_type].append([user_indices[user]] + vector)

	output_folder = vector_path / name	
	output_folder.mkdir(parents=True, exist_ok=True)
	user_interest_sim = {}
	for i_type, vectors in tqdm(interest_vectors.items()):
		if (output_folder / f'{i_type.split("/")[-1]}_similarity.csv').is_file():
		 	continue
		elif vectors == []or len(vectors[0]) == 1:
		 	continue
		logging.info(f'Calculating similarity {name} {i_type}...')
		data = np.array(vectors)
		similarities = pw_cosine_distance(torch.Tensor(data[:, 1:]), torch.Tensor(data[:, 1:])).numpy()
		np.fill_diagonal(similarities, 0)
		similarities *= np.tri(*similarities.shape)
		mean = np.mean(similarities)
		print('mean', mean)
		threshold = np.quantile(similarities, 0.999)
		print('threshold', threshold)
		output = np.vstack((data[:, 0], similarities))
		user_ids = data[:, 0].tolist()
		similarity_flat = {}
		similarity_list = []
		pairs = np.argwhere(similarities >= threshold)
		for user1, user2 in pairs:
			similarity_list.append((user1, user2, similarities[user1, user2]))
			similarity_flat[(user_indices_to_id[user_ids[user1]], user_indices_to_id[user_ids[user2]])] = similarities[user1, user2]

		# for user1 in range(data.shape[0]):
		# 	for user2 in range(data.shape[0]):
		# 		sim = similarities[user1, user2]
		# 		if sim > threshold and user1 != user2:
		# 			similarity_list.append((user1, user2, sim))
		# 			similarity_flat[(user_ids[user1], user_ids[user2])] = sim
		print('pairs', len(similarity_flat))
		user_interest_sim[i_type] = similarity_flat
		
		# pd.DataFrame(similarity_list, columns=['user1', 'user2', 'similarity']).to_csv(output_folder / f'{i_type.split("/")[-1]}_similarity.csv')
		#except:
		#	logging.info(f'No individuals belong to {name} {i_type}...')
		# with open(output_folder / f'{i_type.split("/")[-1]}_similarity.json', 'w') as f:
		# 	f.write(json.dumps(similarity_flat))

		# except:
		# 	logging.info(f'No individuals belong to {name} {i_type}...')



		with open(output_folder / f'{i_type.split("/")[-1]}_users.json', 'w') as f:
			f.write(json.dumps(user_ids))
	interests_final = {}
	records = []
	for i_type, values in user_interest_sim.items():
		for pair, score in values.items():
			if pair not in interests_final:
				interests_final[pair] = {i_type: score}
				records.append((pair[0], pair[1], i_type, score))
			else:
				interests_final[pair][i_type] = score
				records.append((pair[0], pair[1], i_type, score))

	pd.DataFrame(records, columns=['user1', 'user2', 'interest_type', 'score']).to_csv(output_folder / 'interest_similarity.csv')
	# with open(output_folder / 'interest_similarity.json', 'w') as f:
	# 	f.write(json.dumps(interests_final))
