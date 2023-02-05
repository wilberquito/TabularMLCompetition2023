# %%

from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from Util import grid_search_runner
from xgboost import XGBClassifier
import numpy as np
from sklearn.compose import make_column_transformer
from Util import save_model, DropColumns
from sklearn.pipeline import Pipeline

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

base = XGBClassifier(
    random_state=42,
    objective='multi:softmax',
    use_label_encoder=False)

base.fit(X_train, y_train)

echo = '''Train accuracy: \n\t {0}
Test accuracy: \n\t {1}
Parameters: \n\t {2}
'''

print(echo.format(base.score(X_train, y_train),
      base.score(X_test, y_test), base.get_params()))

# %%

print(base.score(X_test, y_test))

# %%

model = XGBClassifier(
    random_state=42,
    objective='multi:softmax',
    use_label_encoder=False,
    n_jobs=-1)

param_grid = {
    'booster': ['gbtree'],
    'alpha': np.arange(0.15, 0.36, 0.05),
    'learning_rate': [0.05, 0.1, 0.2, 0.3],
    'max_depth': [4, 5, 6],
    'n_estimators': np.arange(100, 401, 100, dtype='int'),
}

g_01 = grid_search_runner(model, X_train, y_train,
                          X_test, y_test, param_grid, verbose=2)

# %%

print(g_01.best_estimator_.get_params())

# %%

print(g_01.score(X_test, y_test))

# %%

be = g_01.best_estimator_

columns = list(X.columns)

feature_importance = pd.DataFrame(columns=['importance'],
                                  data=be.feature_importances_,
                                  index=columns) \
    .sort_values(by=['importance'], ascending=False)

print(feature_importance)

# %%

threshold = 0.03
features_to_drop = list(
    feature_importance[feature_importance['importance'] <= threshold].index)

print(f'Features to drop: {features_to_drop}')

# %%

ct = make_column_transformer(
    (DropColumns(features_to_drop), X.columns), remainder='passthrough')

pipe = Pipeline([('transformer', ct), ('model', XGBClassifier(
    random_state=42,
    objective='multi:softmax',
    use_label_encoder=False,
    n_jobs=-1))])

param_grid = {
    'model__booster': ['gbtree'],
    'model__alpha': np.arange(0.15, 0.36, 0.05),
    'model__learning_rate': [0.05, 0.1, 0.2, 0.3],
    'model__max_depth': [4, 5, 6],
    'model__n_estimators': np.arange(100, 401, 100, dtype='int'),
}

g_02 = grid_search_runner(pipe, X_train, y_train, X_test, y_test, param_grid)

# %%

parent_folder = Path('../Model/Base/XGBoost')

# %%

filename = Path('model01.pickle')
save_model(model=g_01, folder=parent_folder, filename=filename)

# %%

filename = Path('model02.pickle')
save_model(model=g_02, folder=parent_folder, filename=filename)

# %%

filename = Path('model03.pickle')
save_model(model=base, folder=parent_folder, filename=filename)
