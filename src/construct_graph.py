import pandas as pd 
from ontology import Ontology

interactions = pd.read_csv('../data/interactions.csv')
entities = pd.read_csv('../data/entities.csv')

mentions = interactions[['UserName', 'MentionedUser', 'weights_mention']].to_records(index=False).tolist()# [:SIZE]
retweets =  interactions[['UserName', 'retweet', 'weights_rt']].to_records(index=False).tolist()# [:SIZE]
ents = entities[["user", "interest type", "interest"]].to_records(index=False).tolist()# [:SIZE]

filtered_mentions = [(user1, user2, weight) for user1, user2, weight in mentions if weight != 0]
filtered_retweets = [(user1, user2, weight) for user1, user2, weight in retweets if weight != 0]
ents_triplets = [(user, types, interests) for user, types, interests in ents]


ontology = Ontology('../interests-v4.owl')
ontology.add_interaction_list(filtered_mentions, 'mention')
ontology.add_interaction_list(filtered_retweets, 'retweet')
ontology.add_entity_list(ents_triplets)
ontology.save('../data/model.rdf')

