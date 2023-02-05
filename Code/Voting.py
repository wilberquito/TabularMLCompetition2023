# %%

from Util import path_to_uuid, power_set
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from Util import find_files
import pickle
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from Util import save_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%

df = pd.read_csv('../Data/preprocessed.csv')
X = df.drop('class', axis=1)
y = df['class']

X.shape, y.shape

# %%

parent_root = '../Model/Base'

# giving file extension
ext = ('.pickle', '.pkl')

pickles = find_files(parent_root, ext)

P = {}

for p in pickles:
    if p.exists():
        with open(p, "rb") as f:
            P[str(path_to_uuid(p))] = pickle.load(f)

# %%

for k, v in P.items():
    if isinstance(v, GridSearchCV):
        P[k] = v.best_estimator_

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)
combinations = power_set(P.items(), min_length=3, max_length=7)
voting_of_models = pd.DataFrame()

for combination in combinations:

    name = list(map(lambda x: x[0], combination))
    name = ('_'.join(name).replace('.pickle', '')) + '.pickle'
    name = name.replace('/', '-')
    print(f'Ensembling by voting: {name}')

    voting = VotingClassifier(estimators=combination, voting='hard', n_jobs=-1)

    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)

    test_score = voting.score(X_test, y_test)

    print(
        f'Classification test score:\n{test_score}')

    entry = {
        'uuid': 'Voting_' + name,
        'model': voting,
        'score': test_score
    }

    voting_of_models = voting_of_models.append(
        entry, ignore_index=True)

# %%

top_N = voting_of_models.sort_values('score', ascending=False).head(10)
folder_voting = Path('../Model/Ensemble/Voting')

for name, score, model in zip(top_N['uuid'], top_N['score'], top_N['model']):
    print(name, score)
    filename = Path(name)
    save_model(model=model, folder=folder_voting, filename=filename)

# %%
