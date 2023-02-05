# %%

from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from Util import grid_search_runner
import numpy as np
from sklearn.compose import make_column_transformer
from Util import save_model, DropColumns
from lightgbm import LGBMClassifier

# %%

df = pd.read_csv('../Data/preprocessed.csv')
df.shape

# %%

X = df.drop('class', axis=1)
y = df['class']

X.shape, y.shape

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%

base = LGBMClassifier(
    random_state=42,
    objective='multi:softmax',
    n_jobs=-1)

base.fit(X_train, y_train)

echo = '''Train accuracy: \n\t {0}
Test accuracy: \n\t {1}
Parameters: \n\t {2}
'''

print(echo.format(base.score(X_train, y_train),
      base.score(X_test, y_test), base.get_params()))

# %%

model = LGBMClassifier(
    random_state=42,
    objective='multi:softmax',
    n_jobs=-1)

param_grid = {
    'learning_rate': np.arange(0.01, 0.3, 0.05),
    'max_depth': [5, 6],
    'bagging_fraction': [0.5],
    'bagging_freq': [3],
    'colsample_bytree': [0.7, 0.8],
    'n_estimators': np.arange(100, 301, 50, dtype='int'),
}
# %%

model = LGBMClassifier(
    random_state=42,
    objective='multi:softmax',
    n_jobs=-1)

param_grid = {
    'learning_rate': np.arange(0.01, 0.07, 0.01),
    'max_depth': [5, 6],
    'bagging_fraction': [0.5],
    'bagging_freq': [3],
    'colsample_bytree': [0.7, 0.8],
    'n_estimators': np.arange(100, 301, 50, dtype='int'),
}

g_01 = grid_search_runner(model, X_train, y_train, X_test, y_test, param_grid, verbose=10)


# %%

model = LGBMClassifier(
    random_state=42,
    objective='multi:softmax',
    n_jobs=-1)

param_grid = {
    'learning_rate': np.arange(0.001, 0.01, 0.002),
    'num_leaves': np.arange(10, 31, 5),
    'n_estimators': np.arange(100, 401, 100, dtype='int'),
    'max_depth': [4, 5, 6],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0.1, 0.2],
    'reg_lambda': [0.1, 0.2]
}

g_01 = grid_search_runner(model, X_train, y_train, X_test, y_test, param_grid)

# %%

be = g_01.best_estimator_

columns = list(X.columns)

feature_importance = pd.DataFrame(columns=['importance'],
                                  data=be.feature_importances_,
                                  index=columns) \
    .sort_values(by=['importance'], ascending=False)

print(feature_importance)
features_to_drop = ['man']

# %%

ct = make_column_transformer(
    (DropColumns(features_to_drop), X.columns), remainder='passthrough')

param_grid = {
    'learning_rate': np.arange(0.001, 0.01, 0.002),
    'num_leaves': np.arange(10, 31, 5),
    'n_estimators': np.arange(100, 301, 100, dtype='int'),
    'max_depth': [4, 5, 6],
}

g_02 = grid_search_runner(model, X_train, y_train,
                          X_test, y_test, param_grid, verbose=2)

# %%

parent_folder = Path('../Model/Base/LGBMClassifier')

# %%

filename = Path('model01.pickle')
save_model(model=g_01, folder=parent_folder, filename=filename)

# %%

filename = Path('model02.pickle')
save_model(model=g_02, folder=parent_folder, filename=filename)

# %%

filename = Path('model03.pickle')
save_model(model=base, folder=parent_folder, filename=filename)
