import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import col_transformer


# Import data
X_eval = np.load('../data/X_eval400.npy', allow_pickle=True)
X_public = np.load('../data/X_public400.npy', allow_pickle=True)
y_public = np.load('../data/y_public400.npy', allow_pickle=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_public, y_public,
                                                    test_size=0.2, random_state=1, stratify=y_public)

# Preprocessing
X_train = col_transformer.fit_transform(X_train)
X_test = col_transformer.transform(X_test)

model_params = {
    'svc': {
        'model': SVC(),
        'params': {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': np.logspace(-3, 3, 7),
            'degree': np.arange(1, 6),
            'coef0': np.logspace(-3, 3, 7),
            'gamma': np.logspace(-3, 3, 7)
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(),
        'params': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': np.logspace(-3, 3, 7),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
    },
    'decision_trees': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
            'min_samples_split': [2, 3, 4]
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': np.logspace(0, -9, num=100)
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': np.arange(1, 20),
            'weights': ['uniform', 'distance']
        }
    }
    # 'linear_svc': {
    #     'model': LinearSVC(),
    #     'params': {
    #         'penalty': ['l1', 'l2'],
    #         'loss': ['hinge', 'squared_hinge'],
    #         'dual': [True, False],
    #         'C': [0.0001, 0.0005, 0.001, 0.01, 0.1, 10, 100]
    #
    #     }
    # },
    # 'nu_svc': {
    #     'model': NuSVC(),
    #     'params': {
    #         'nu': np.arange(0.1, 1, 0.1),
    #         'kernel': ['linear', 'poly', 'rbf'],
    #         'degree': np.arange(1, 6),
    #         'tol':  np.logspace(-3, 3, 7),
    #         'probability': [True, False]
    #     }
    # }
}
#
scores = []
start = time.time()

for model_name, mp in model_params.items():
    # gs = GridSearchCV(mp['model'], mp['params'], cv=3, n_jobs=-1, verbose=1)
    gs = RandomizedSearchCV(mp['model'], mp['params'], cv=3, n_jobs=-1, n_iter=100, verbose=1)
    gs.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': gs.best_score_,
        'best_params': gs.best_params_
    })

stop = time.time()

print(f"Training time: {stop - start}s")
scores_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
