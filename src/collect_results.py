import pandas as pd
import json
from pathlib import Path

size = 50000
algorithm = 'rb_pots'
dfs = []
path = Path(f'../results_{size}/{algorithm}/')
for folder in path.iterdir():
	if folder.is_dir():
		for model in folder.iterdir():
			scores = json.loads(open(model / 'scores.json').read())
			dfs.append({'model': model.name, 'level': folder.name, **scores})

df = pd.DataFrame(dfs)
df.to_csv(f'../results_{size}/results.csv')
