import pandas as pd 
from ontology import Ontology
interactions = pd.read_csv('../data/interactions.csv')

mentions = interactions[['UserName', 'MentionedUser', 'weights_mention']].to_records(index=False).tolist()
retweets =  interactions[['UserName', 'retweet', 'weights_rt']].to_records(index=False).tolist()

filtered_mentions = [(user1, user2, weight) for user1, user2, weight in mentions if weight != 0]
filtered_retweets = [(user1, user2, weight) for user1, user2, weight in retweets if weight != 0]


ontology = Ontology('../interests-v3.owl')
ontology.add_interaction_list(filtered_mentions, 'mention')
ontology.add_interaction_list(filtered_retweets, 'retweet')
ontology.save('../data/populated_interests.owl')

