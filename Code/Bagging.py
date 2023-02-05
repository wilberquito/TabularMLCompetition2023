# Load libraries
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd

# Load libraries
from sklearn.model_selection import GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
import pandas as pd
from Util import grid_search_runner

# %%

df = pd.read_csv('../Data/preprocessed.csv')
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%

abc = AdaBoostClassifier(n_estimators=1000,
                         learning_rate=1,
                         random_state=42)
# Train Adaboost Classifer
base = abc.fit(X_train, y_train)

echo = '''Train accuracy: \n\t {0}
Test accuracy: \n\t {1}
Parameters: \n\t {2}
'''.format(base.score(X_train, y_train),
           base.score(X_test, y_test), base.get_params())

print(echo)

# %%

import numpy as np

ct = make_column_transformer(
    (StandardScaler(), X.columns), remainder='passthrough')

pipe = Pipeline([('transformer', ct), ('svc', SVC(probability=True))])

params_grid = {
    'svc__C': [0.1, 1, 5, 10, 20],
    'svc__kernel': ['poly', 'rbf'],
    'svc__degree': [3, 4],
    'svc__coef0': np.arange(0, 1.2, 0.2)
}

g_01 = grid_search_runner(pipe, X_train, y_train,
                          X_test, y_test, params_grid, verbose=2)


# Create adaboost classifer object
abc = AdaBoostClassifier(
    n_estimators=50, base_estimator=g_01.best_estimator_, learning_rate=1, random_state=42)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
