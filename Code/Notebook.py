# %%

from sklearn import metrics
from Util import find_files, path_to_uuid
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
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

predictors = []
names = []
precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
train_accuracy_score = []

for uuid, model in P.items():

    if hasattr(model, 'best_estimator_'):
        predictor = model.best_estimator_
    else:
        predictor = model

    columns = X.columns

    X_model = X[columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, random_state=42, test_size=0.2, stratify=y)

    y_pred = predictor.predict(X_test)

    names.append(uuid)
    predictors.append(predictor)
    precision_score.append(round(metrics.precision_score(
        y_test, y_pred, average='weighted'), 5))
    recall_score.append(round(metrics.recall_score(
        y_test, y_pred, average='weighted'), 5))
    f1_score.append(round(metrics.f1_score(
        y_test, y_pred, average='weighted'), 5))
    accuracy_score.append(round(metrics.accuracy_score(y_test, y_pred), 5))
    train_accuracy_score.append(round(model.score(X_train, y_train), 5))

# %%


data = {
    'precision_score':  precision_score,
    'recall_score': recall_score,
    'f1_score': f1_score,
    'accuracy_score': accuracy_score,
    'train_accuracy_score': train_accuracy_score,
    'predictor': predictors
}

df_data = pd.DataFrame(data, index=names)
df_data = df_data.sort_values('accuracy_score', ascending=False).head(10)

for i, row in df_data.iterrows():
    print(i, row['accuracy_score'], row['train_accuracy_score'])

# %%

stats = df_data.drop('predictor', axis=1)
stats['uuid'] = stats.index
stats.to_csv('metrics.csv', index=True)
print(stats.head(10))

# %%

X_base = pd.read_csv('../Data/validate.csv')
gender = X_base['gender']
gender_ohc = OneHotEncoder(drop='first')
gender = gender_ohc.fit_transform(gender.values.reshape(-1, 1))
X_base['man'] = pd.Series(gender.toarray().squeeze())
X_base = X_base.drop('gender', axis=1, errors='ignore')
X_val = X_base.drop('id', axis=1)

# %%

Path('../Submit').mkdir(parents=True, exist_ok=True)

label_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

for i, row in df_data.iterrows():
    submit_path = i.split('/')[-1]
    submit_path = submit_path.split('.')[-2] + '.csv'
    submit_path = Path('../Submit') / Path(submit_path)

    y_pred = predictor.predict(X_val)

    submit = pd.DataFrame({
        'id': X_base['id'],
        'class': pd.Series(y_pred).apply(lambda x: label_mapping[x])
    })

    submit.to_csv(submit_path, index=False)

# %%
