from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from itertools import islice
from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import preprocessor as p


def translate(s):
    return ''.join(ch for ch in str(s) if ch.isalnum())

@st.cache(allow_output_mutation=True)
def get_data(alpha1, alpha2, alpha3):
	size= 25000
	algorithm = 'rb_pots'
	interest_level = 'subclass'
	# alpha1 = 1
	# alpha2 = 0
	# alpha3 = 0
	path = Path(f'../results_{size}/{algorithm}/{interest_level}/weights_{alpha1}_{alpha2}_{alpha3}')
	data = json.loads(open(path / 'predictions.json').read())
	id_to_node = data['id_node']
	communities = {}
	for cid, community in enumerate(data['communities']):
		for user in community[:(len(community) // 2)]:
			label = id_to_node[str(user)].split('#')[-1]
			communities[label] = cid
	db_tweets = pd.read_csv('../data/db_annotated_tweets.csv')
	db_tweets['UserName'] = db_tweets['UserName'].apply(lambda x: translate(str(x).replace('@', '').replace(' ', '')))
	db_tweets['Community'] = db_tweets['UserName'].apply(lambda x: communities.get(x, -1))
	db_tweets['CleanTweet'] = db_tweets['TweetText'].apply(p.clean)
	return db_tweets, data

st.title('Community Terms')
st.markdown('### Select alpha values')

alpha1 = st.selectbox('Select alpha1',  [1, -1, 0])
alpha2 = st.selectbox('Select alpha2',  [1, -1, 0])
alpha3 = st.selectbox('Select alpha3',  [1, -1, 0])

db_tweets, data = get_data(alpha1, alpha2, alpha3)
st.markdown('### Choose a community')

print(db_tweets['Community'].value_counts())


option = st.selectbox( 'Select community id',  list(range(len(data['communities']))))

st.write('You selected community:', option)
# for i in range(10):
tweets = db_tweets[db_tweets['Community'] == option]
cvec = CountVectorizer(stop_words='english', min_df=5, max_df=0.95, ngram_range=(1,2))
cvec.fit(tweets['CleanTweet'])
cvec_count = cvec.transform(tweets['CleanTweet'])
print('Sparse Matrix Shape : ', cvec_count.shape)
print('Non Zero Count : ', cvec_count.nnz)
print('sparsity: %.2f%%' % (100.0 * cvec_count.nnz / (cvec_count.shape[0] * cvec_count.shape[1])))

occ = np.asarray(cvec_count.sum(axis=0)).ravel().tolist()
count_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences' : occ})

topn = st.slider('Number of terms', 20, 100, 10)

term_freq = count_df.sort_values(by='occurrences', ascending=False).head(topn)
# print(term_freq)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cvec_count)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weight_df = pd.DataFrame({'term' : cvec.get_feature_names(), 'weight' : weights})
tf_idf = weight_df.sort_values(by='weight', ascending=False).head(topn)
print(tf_idf)
fig = px.bar(tf_idf, y='term', x='weight', orientation='h')
st.plotly_chart(fig, use_container_width=True)
