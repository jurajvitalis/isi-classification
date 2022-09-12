import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import col_transformer


# Import data
X_eval = np.load('../data/X_eval400.npy', allow_pickle=True)
X_public = np.load('../data/X_public400.npy', allow_pickle=True)
y_public = np.load('../data/y_public400.npy', allow_pickle=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_public, y_public,
                                                    test_size=0.2, random_state=1, stratify=y_public)

# Create the main pipeline
pipe = Pipeline([
    ('preprocessing', col_transformer),
    ('clf', SVC())
])

# Grid search
param_grid = {
    'clf__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
    'clf__C': np.logspace(-3, 3, 7),
    'clf__degree': np.arange(1, 6),
    'clf__coef0': np.logspace(-3, 3, 7),
    'clf__gamma': np.logspace(-3, 3, 7)
}
gs = GridSearchCV(pipe, param_grid, cv=3, verbose=1, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)

# Output
print(f'Best params: {gs.best_params_}')
print(f'Best score: {gs.best_score_}')
df = pd.DataFrame(gs.cv_results_)
joblib.dump(gs, 'gs_result.pkl')
