import preprocessor as p
import spacy_dbpedia_spotlight
import spacy
import pandas as pd
import json
import re
import argparse
import logging

from ast import literal_eval
from tqdm import tqdm
from joblib import Parallel, delayed
tqdm.pandas()


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class TweetProcessor:

	def __init__(self):

		self.dbpedia_nlp = spacy.load('en_core_web_sm')
		self.dbpedia_nlp.add_pipe('dbpedia_spotlight')

		self.wikidata_nlp = spacy.load('en_core_web_sm')
		self.wikidata_nlp.add_pipe('opentapioca')


	def process(self, tweet):
		try: 
			m = re.search(r'RT\s.([^@].*?):', tweet['TweetText'])
			tweet['retweet'] = m.group(1) if m else ''

			text = p.clean(tweet['TweetText'])
			parsed_tweet = p.parse(tweet['TweetText'])


			if tweet['collector'] == 'scraper':
				tweet['TweetID'] = tweet['Tweet URL'].split('/')[-1]
				if parsed_tweet.mentions:
					tweet['mentions'] = ';'.join([f'{m.match[1:]},' for m in parsed_tweet.mentions])
				else:
					tweet['mentions'] = ''
			elif tweet['collector'] == 'api':
				entities = literal_eval(tweet['Entities'])
				if entities:
					tweet['mentions'] = ';'.join([f'{mention["screen_name"]},{mention["id_str"]}' for mention in entities['user_mentions']])
				else:
					tweet['mentions'] = ''

			dbpedia_ent = self.dbpedia_nlp(text)
			wd_ent = self.wikidata_nlp(text)
			db_ents = [(ent.text, ent.label_, ent.kb_id_, ent._.description, ent._.dbpedia_raw_result['@similarityScore'] if ent._.dbpedia_raw_result is not None else '', ent._.dbpedia_raw_result['@types'] if ent._.dbpedia_raw_result is not None else '') 
				for ent in dbpedia_ent.ents if ent.label_ not in ['CARDINAL', 'PERCENT']]
			tweet['db_ents'] = ';'.join([','.join([f'"{item}"' if item is not None else '' for item in ent]) for ent in db_ents])
			wd_ents = [(ent.text, ent.label_, ent.kb_id_, ent._.description, ent._.score) for ent in wd_ent.ents if ent.label_ not in ['CARDINAL', 'PERCENT']]
			tweet['wd_ents'] = ';'.join([','.join([f'"{item}"' if item is not None else '' for item in ent]) for ent in wd_ents])
		except Exception as e:
			print(e)
		return tweet

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str)
	parser.add_argument('--output', type=str)
	args = parser.parse_args()
	logging.info(f'Processing tweets from {args.input}, output file: {args.output}')
	processor = TweetProcessor()

	# annotated_tweets = all_tweets.swifter.progress_bar(True).apply(processor.process, axis=1)
	# annotated_tweets.to_csv('../data/annotated_tweets.csv')
	# tweet_records = tweets.to_dict('records')
	# annotated_tweets = [processor.process(record) for record in tqdm(tweet_records)]
	# annotated_tweets = Parallel(n_jobs=5)(delayed(processor.process)(record) )

	tweets = pd.read_csv(args.input).reset_index()[:100]
	tweet_records = tweets.to_dict('records')
	print(tweet_records[:3])
	# annotated_tweets = tweets.progress_apply(processor.process, axis=1)
	annotated_tweets = Parallel(n_jobs=10, verbose=10)(delayed(processor.process)(record) for record in tqdm(tweet_records))
	pd.DataFrame(annotated_tweets).to_csv(args.output)
