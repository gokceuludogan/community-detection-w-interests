import pandas as pd
import subprocess
from pathlib import Path
from joblib import Parallel, delayed


def run_processor(nput_file, output_file):
	returned_bin = subprocess.check_output(f'python processor.py --input {input_file} --output {output_file}'.split())


tweets = pd.read_csv('../data/tweets.csv')

chunk_dir = Path('../data/chunks_v2/')
chunk_dir.mkdir(exist_ok=True)

chunk_size = 1000 # int(tweets.shape[0] / 5)
files = []
for start in range(0, tweets.shape[0], chunk_size):
    df_subset = tweets.iloc[start:start + chunk_size]
    input_file = str(chunk_dir / f'chunk{start}.csv')
    df_subset.to_csv(input_file)
    files.append((input_file, str(chunk_dir / f'annotated_chunk{start}.csv')))

Parallel(n_jobs=12)(delayed(run_processor)(input_file, output_file) for input_file, output_file in files)  

annotations = []
for file in chunk_dir.iterdir():
	if file.name.startswith('annotated_chunk'):
		annotations.append(pd.read_csv(file))

pd.concat(annotations).to_csv('../data/annotated_tweets.csv')