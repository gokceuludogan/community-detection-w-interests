import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path 

def merge_json(directory, output_file):
	tweets = []
	user_fields = ['id_str', 'screen_name', 'followers_count', 'friends_count', 'favourites_count', 'created_at']
	data_dir = Path(directory)
	for file in tqdm(data_dir.iterdir()):
		if file.is_file() and file.name.endswith('json'):
			tweet_data = json.loads(open(file).read())
			filtered = {k:v for k, v in tweet_data.items() if k != 'user'}
			user_data = {f'user_{item}': tweet_data['user'][item] for item in user_fields}
			tweets.append({**filtered, **user_data})

	pd.DataFrame(tweets).to_csv(data_dir.parent / output_file)

def merge_csv(directory, output_file):
	tweets = []
	data_dir = Path(directory)
	for file in tqdm(data_dir.iterdir()):
		if file.is_file() and file.name.startswith('tweets_scweet'):
			tweets.append(pd.read_csv(file))

	pd.concat(tweets).to_csv(data_dir / output_file)


def merge_users(directory, output_file, threshold=3):
	user_list = []
	data_dir = Path(directory)
	for file in data_dir.iterdir():
		if file.name.endswith('.csv') and file.name.startswith('tweets'):
			df = pd.read_csv(file)
			try: 
				users = df['user_screen_name'].value_counts()
				print(users)
				print(users[users >=  threshold])
				user_list.append(users[users >=  threshold])
			except:
				users = df['UserScreenName'].value_counts()
				print(users)
				print(users[users >=  threshold])
				user_list.append(users[users >=  threshold])

	pd.concat(user_list).to_csv(data_dir / output_file)

if __name__ == '__main__':
	merge_json('../data/after_dec3', 'tweets_after_dec3.csv')
	merge_csv('../data/before_dec3', 'tweets_till_dec3.csv')
	merge_users('../data', 'user_screen_names.csv')
	
	df1_cols = {'UserScreenName': 'UserName', 'Embedded_text': 'TweetText', 'Timestamp': 'Timestamp', 'Tweet URL': 'Tweet URL'}
	df2_cols = {'created_at': 'Timestamp', 'id_str': 'TweetID', 'full_text': 'TweetText', 'entities': 'Entities', 'user_id_str': 'UserID', 'user_screen_name': 'UserName'}

	df1 = pd.read_csv('../data/tweets_till_dec3.csv')
	df1 = df1.rename(columns=df1_cols)[df1_cols.values()].assign(collector='scraper')

	df2 = pd.read_csv('../data/tweets_after_dec3.csv')
	df2 = df2.rename(columns=df2_cols)[df2_cols.values()].assign(collector='api')


	all_tweets = pd.DataFrame(df1.to_dict('records') + df2.to_dict('records'))
	all_tweets.to_csv('../data/tweets.csv')
	