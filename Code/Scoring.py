# %%

from Util import find_files, path_to_uuid
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
pd.set_option('display.max_columns', None)

# %%

df = pd.read_csv('../Data/preprocessed.csv')
X = df.drop('class', axis=1)
y = df['class']

X.shape, y.shape

# %%

dirname = '../Model'

# giving file extension
ext = ('.pickle', '.pkl')

pickles = find_files(dirname, ext)

print(pickles)

P = {}

for p in pickles:
    if p.exists():
        with open(p, "rb") as f:
            P[str(path_to_uuid(p))] = pickle.load(f)


# %%

models = []
scores = []

for uuid, model in P.items():

    if hasattr(model, 'best_estimator_'):
        predictor = model.best_estimator_
    else:
        predictor = model

    columns = X.columns

    X_model = X[columns]

    _, X_test, _, y_test = train_test_split(
        X_model, y, random_state=42, test_size=0.2, stratify=y)

    y_pred = predictor.predict(X_test)

    models.append(uuid)
    model_report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True))
    scores.append(model.score(X_test, y_test))

# %%

models_data = pd.DataFrame(
    {
        'model': models,
        'score': scores
    }
)

models_data = models_data.sort_values('score', ascending=False)

for _, row in models_data.iterrows():
    print(row['model'], row['score'])
