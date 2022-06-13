import numpy as np
import pandas as df
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from clf_preprocessing import X_train_final, y_train, X_test_final, y_test


model_params = {
    'svc': {
        'model': SVC(),
        'params': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 10, 100, 1000, 10000],
            'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'coef0': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100]
        }
    },
    # 'linear_svc': {
    #     'model': LinearSVC(),
    #     'params': {
    #         'penalty': ['l1', 'l2'],
    #         'loss': ['hinge', 'squared_hinge'],
    #         'dual': [True, False],
    #         'C': [0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 10, 100, 1000, 10000]
    #
    #     }
    # }
    # 'nu_svc': {
    #     'model': NuSVC(),
    #     'params': {
    #         'nu': [np.arange(0.1, 1, 0.1)],
    #         'kernel': ['linear', 'poly', 'rbf'],
    #         'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'tol': [0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 10, 100, 1000, 10000],
    #         'probability': [True, False]
    #     }
    # }
}

scores = []
for model_name, mp in model_params.items():
    print('BEGIN')
    print(f'{model_name}, {mp}')
    clf = GridSearchCV(mp['model'], mp['params'], cv=10, scoring='roc_auc', return_train_score=False, n_jobs=2)
    clf.fit(X_train_final, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    print('END')

scores_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print('FINISHED')
