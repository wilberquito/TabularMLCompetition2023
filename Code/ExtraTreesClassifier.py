
# %%

from sklearn.compose import make_column_transformer
from Util import save_model, DropColumns, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from Util import grid_search_runner
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

# Base Random Forest
from sklearn.ensemble import ExtraTreesClassifier

base = ExtraTreesClassifier(n_jobs=-1, random_state=42, bootstrap=True)
base.fit(X_train, y_train)

echo = '''Train accuracy: \n\t {0}
Test accuracy: \n\t {1}
Parameters: \n\t {2}
'''

print(echo.format(base.score(X_train, y_train),
      base.score(X_test, y_test), base.get_params()))

plot_confusion_matrix(y_test, base.predict(X_test), [0, 1, 2, 3])

# %%

param_grid = {'n_estimators': [50, 100, 250, 500, 1000],
              'max_depth': [6, 8, 12, 13, 14],
              'min_samples_leaf': [2, 3, 4],
              'criterion': ['gini']}

clf = RandomForestClassifier(n_jobs=-1, random_state=42)

g_01 = grid_search_runner(clf, X_train, y_train, X_test, y_test, param_grid)

# %%

print(g_01.best_estimator_.score(X_test, y_test))
print(g_01.best_estimator_.get_params())
plot_confusion_matrix(y_test, g_01.predict(X_test), [0, 1, 2, 3])

# %%

be = g_01.best_estimator_

feature_importance = pd.DataFrame(columns=['importance'],
                                  data=be.feature_importances_,
                                  index=be.feature_names_in_) \
    .sort_values(by=['importance'], ascending=False)

print(feature_importance)

# %%

threshold = 0.02
features_to_drop = list(
    feature_importance[feature_importance['importance'] <= threshold].index)

print(f'Features to drop: {features_to_drop}')

# %%

ct = make_column_transformer(
    (DropColumns(features_to_drop), X.columns), remainder='passthrough')

pipe = Pipeline([('transformer', ct), ('model',
                RandomForestClassifier(n_jobs=-1, random_state=42))])

param_grid = {'model__n_estimators': [50, 100, 250, 500, 1000],
              'model__max_depth': [6, 8, 12, 13, 14],
              'model__min_samples_leaf': [2, 3, 4],
              'model__criterion': ['gini']}

g_02 = grid_search_runner(pipe, X_train, y_train, X_test, y_test, param_grid)

# %%

folder_name = Path('../Model/Base/RandomForest')

# %%

filename = Path('model01.pickle')
save_model(model=g_01, folder=folder_name, filename=filename)

# %%

folder = Path('../Models/RandomForest')
RandomForestClassifier(n_jobs=-1, random_state=42)
filename = Path('model02.pickle')
save_model(model=g_02, folder=folder_name, filename=filename)

# %%

folder = Path('../Models/RandomForest')
RandomForestClassifier(n_jobs=-1, random_state=42)
filename = Path('model03.pickle')
save_model(model=base, folder=folder_name, filename=filename)
