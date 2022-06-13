import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from old.data_preprocessing import X_train_final, y_train

param = {
       'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
       'C': np.arange(0.0001, 1000, 10),
       'degree': np.arange(2, 6),
       'coef0': np.arange(0.001, 3, 0.5),
       'gamma': ('auto', 'scale')
}

grid = GridSearchCV(SVC(), param, refit=True, verbose=10, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_final, y_train)

print(grid.best_params_)
print(grid.best_score_)
