import tweepy
import json
import pandas as pd
import random
import logging
import argparse
from datetime import timedelta, date
from Scweet.scweet import scrape
from joblib import Parallel, delayed
from pathlib import Path

consumer_key = 'fUUC4KcibC0Ibezb8Yv4vFxkf'
consumer_secret = '9ycumDsraLUHOxVylNM51YP975I904v81Lz0LAnSpsrlxJLUv7'
access_token = '757556294110838784-GzqG678snZoj6yZGk6ugEJijkWtFGEx'
access_token_secret = 'ALPdSW0373iLUZ3Kfhx62LU2dVLBD1I6aDJA0xwgQjP0O'

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class TweetDownloader:

	def __init__(self):
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		self.api = tweepy.API(auth, wait_on_rate_limit=True)

	def search_tweets(self, keywords, output_dir, until=None):
		counter = 0
		output = Path(output_dir)
		output.mkdir(exist_ok=True, parents=True)

		for tweet in tweepy.Cursor(self.api.search_tweets, keywords,
									count=100, until=until, tweet_mode='extended').items():
			if counter % 1000 == 0:
				logging.info(f'Downloaded {str(counter)} tweets')
			with open(output / (tweet.id_str + '.json'), 'w') as f: 
				f.write(json.dumps(tweet._json))
			counter += 1


	def user_timeline(self, screen_name, output_dir, user_id=None):
		counter = 0
		output = Path(output_dir) / screen_name
		output.mkdir(exist_ok=True, parents=True)

		for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=screen_name, since_id='1452424849641508870', tweet_mode="extended").items():
			if counter % 1000 == 0:
				logging.info(f'Downloaded {str(counter)} tweets')
			with open(output / (tweet.id_str + '.json'), 'w') as f: 
				f.write(json.dumps(tweet._json))
			counter += 1

	def download_users(self, users, user_dir):
		random.shuffle(user_list)
		user_dir = Path(user_dir)
		for user in user_list:
			try:
				if not (user_dir / user).is_dir():
					logging.info(f'Downloading tweets {user}')
					downloader.user_timeline(user, str(user_dir))
			except Exception as e:
				logging.error(e)


def daterange(date1, date2):
	for n in range(int ((date2 - date1).days)+1):
		yield date1 + timedelta(n)

def scrape_interval(words, interval, output_dir):
	try:
		logging.info(f'scraping {interval}')
		data = scrape(words=words, since=interval[0], until=interval[1], from_account = None,         
			interval=1, headless=True, display_type="Latest", save_images=False, lang="en",
			resume=False, filter_replies=False, proximity=False)
		data.to_csv(f'{output_dir}/tweets_scweet_{interval[0]}_{interval[1]}.csv')

	except:
		logging.error(f'Error scraping {interval}')

def generate_intervals(start_date, end_date, directory):
	start_date = map(int, start_date.split('-'))
	end_date = map(int, end_date.split('-'))
	start_dt = date(start_date[0], start_date[1], start_date[2])
	start_dt2 = start_dt + datetime.timedelta(days=1)
	end_dt = date(end_date[0], end_date[1], end_date[2])

	intervals = []
	files = [file.name for file in Path(directory).iterdir()]
	for dt in daterange(start_dt2, end_dt):
		since = start_dt.strftime("%Y-%m-%d")
		until = dt.strftime("%Y-%m-%d")
		if f'tweets_scweet_{since}_{until}.csv' not in files:
			intervals.append((since, until))
		else:
			logging.info(f'tweets_scweet_{since}_{until}.csv already downloaded.')
		start_dt = dt
	return intervals

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, choices=['api', 'scraper'])
	parser.add_argument('--data', type=str, choices=['keyword', 'user'])

	args = parser.parse_args()

	if args.method == 'api':
		downloader = TweetDownloader()
		if args.data == 'keyword':
			downloader.search_tweets('rittenhouse OR #rittenhousetrial OR #kylerittenhouse OR Rittenhouse',
				output_dir='../data/after_dec3', until='2021-12-03')
		elif args.data == 'user':
			df = pd.read_csv('../data/user_screen_names.csv')
			df.columns = ['screen_name', 'tweet_count']
			user_list = df['screen_name'].tolist()
			downloader.download_users(user_list, '../data/user_data/')

	elif args.method == 'scraper':
		words = ['rittenhouse', 'rittenhousetrial']
		intervals = generate_intervals('2021-10-25', '2021-12-03', '../data/before_dec3')
		Parallel(n_jobs=5)(delayed(scrape_interval)(words, i, '../data/before_dec3') for i in intervals)


