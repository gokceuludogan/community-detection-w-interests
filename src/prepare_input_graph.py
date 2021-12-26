import json
import time
import logging
from pathlib import Path
from ontology import Ontology
from tqdm import tqdm

def compute_intersection(interests1, interests2):
	classes = set(interests1.keys()).union(set(interests2.keys()))
	results = {}
	for class_ in classes: 
		set1 = set(interests1.get(class_, []))
		set2 = set(interests2.get(class_, []))
		results[class_] = {'common': set1.intersection(set2), 'count': set1.union(set2)}
	return results

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')


logging.info('Loading ontology...')
ontology = Ontology('../data/model.rdf') # sample.owl')

logging.info('Retrieving mentions...')
mentions = ontology.get_all_interactions('mention')

logging.info('Retrieving retweets...')
retweets = ontology.get_all_interactions('retweet')

mention_dict = {(user1, user2): weight for user1, user2, weight in mentions}
retweet_dict = {(user1, user2): weight for user1, user2, weight in retweets}

logging.info('Retrieving interests...')

interests = ontology.get_all_interests()

user_interests = {}
for user, interest, interest_type in interests:
	if user not in user_interests:
		user_interests[user] = {interest_type: [interest]}
	else:
		if interest_type not in user_interests[user]:
			user_interests[user][interest_type] = [interest]
		else:
			user_interests[user][interest_type].append(interest)

logging.info('Finding user interest similarities')

interest_dict = {}
for user1, class_interests1 in tqdm(user_interests.items()):
	for user2, class_interests2 in user_interests.items():
		if user1 != user2:
			interest_dict[(user1, user2)] = compute_intersection(class_interests1, class_interests2)

interest_user_sim = {}
for user1, user2 in interest_dict:
	counts = interest_dict[(user1, user2)]
	interest_user_sim[(user1, user2)] = [counts[interest_type]['common'] / counts[interest_type]['count'] for interest_type in counts]	
	

logging.info('Constructing homogenoues graph')
user_interest_dict = {pair: sum(class_sim)/len(class_sim) for pair, class_sim in interest_user_sim.items()}

all_pairs = set(mention_dict).union(set(retweet_dict)).union(set(user_interest_dict))

data = json.dumps({key: {'mention': mention_dict.get(key, 0), 'retweet': retweet_dict.get(key, 0), 'interest': user_interest_dict.get(key, 0)} for key in all_pairs})
with open('../data/user_similarities.json', 'w') as f:
	f.write(data)
