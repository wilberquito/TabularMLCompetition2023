# %%

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from Util import save_model
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from Util import grid_search_runner
import numpy as np

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

ct = ColumnTransformer(
    [('scaler', StandardScaler(), X.columns)], remainder='passthrough')

pipe = Pipeline([('transformer', ct), ('svc', SVC(probability=False))])

params_grid = {
    'svc__C': np.arange(19, 23, 1),
    'svc__kernel': ['poly'],
    'svc__degree': [3],
    'svc__coef0': np.arange(0.5, 1.3, 0.1)
}

g_01 = grid_search_runner(pipe, X_train, y_train,
                          X_test, y_test, params_grid, verbose=2)

# %%

print(g_01.best_estimator_[-1].get_params())

# %%

parent_folder = Path('../Model/Base/SVM')

# %%

filename = Path('model01.pickle')
save_model(model=g_01, folder=parent_folder, filename=filename)


# %%
