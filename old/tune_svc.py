import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from old.data_preprocessing import X_train_final, y_train

clf = SVC()

svc_params = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 10, 25, 50, 75, 100, 250, 500, 1000, 10000],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'coef0': [-1, 0, 1]
}

print('BEGIN')
gs = GridSearchCV(clf, svc_params, cv=10, verbose=10, scoring='roc_auc', return_train_score=False)
gs.fit(X_train_final, y_train)

print('END')
print(f'Best estimator: {gs.best_estimator_}')
print(f'Best score: {gs.best_score_}')
print(f'Best params: {gs.best_params_}')

gs_df = pd.DataFrame(gs.cv_results_)
