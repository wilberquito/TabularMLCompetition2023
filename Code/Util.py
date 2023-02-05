from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from more_itertools import powerset
from collections.abc import Iterable


def path_to_uuid(path: Path):
    return '/'.join(path.parts[-2:])


def save_model(model, folder: Path, filename: Path):

    path = folder / filename

    if (path.exists()):
        os.remove(path)

    folder.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

def power_set(items: Iterable, min_length : int = 0, max_length: int = -1) -> list:

    max_length = min_length if max_length < 0 else max_length

    list_of_tuples = list(powerset(items))
    list_of_lists = [list(elem) for elem in list_of_tuples]

    return [list for list in list_of_lists if len(list)>=min_length and len(list) <= max_length]

def grid_search_runner(model, X_train, y_train, X_test, y_test, grid, cv=5, verbose=0):

    grid_search = GridSearchCV(
        model, grid, cv=cv, n_jobs=-1, verbose=verbose)
    grid_search.fit(X_train, y_train)

    be = grid_search.best_estimator_

    echo = '''Best cv accuracy: \n\t {0}
    Train accuracy: \n\t {1}
    Test accuracy: \n\t {2}
    Parameters: \n\t {3}
    '''

    print(echo.format(grid_search.best_score_, be.score(
        X_train, y_train), be.score(X_test, y_test), be.get_params()))

    return grid_search


def find_files(directory, extensions):
    matches = []
    for root, _, files in os.walk(directory):
        for file in files:
            check = [file.endswith(e) for e in extensions]
            if any(check):
                matches.append(Path(os.path.join(root, file)))
    return matches


class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns
        self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the DataFrame has the columns to be dropped
        X = X.copy()
        for column in self.columns:
            if column in X.columns:
                X = X.drop(column, axis=1)
        self.feature_names = X.columns
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    disp.plot()
    plt.show()
