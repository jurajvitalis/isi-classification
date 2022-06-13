import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import time


# LOAD DATA
X_eval = np.load('../data/X_eval200.npy', allow_pickle=True)
X_public = np.load('../data/X_public200.npy', allow_pickle=True)
y_public = np.load('../data/y_public200.npy', allow_pickle=True)

# OHE
# First, use OHE on X_public
ohe = OneHotEncoder(handle_unknown='ignore')
# Slice the columns containing nominal values
X_public_nominal = X_public[:, 180:201]
# Train the OHE instance
ohe.fit(X_public_nominal)
# Create dummy columns
ohe_dummies = ohe.transform(X_public_nominal).toarray()
# Drop columns containing nominal values
X_public = np.delete(X_public, np.s_[180:201], axis=1)
# Replace them with dummy columns from OHE
X_public = np.hstack((X_public, ohe_dummies[:, 0:180]))
# Split the public dataset
X_train, X_test, y_train, y_test = train_test_split(X_public, y_public,
                                                    test_size=0.2, random_state=0, stratify=y_public)

# PIPELINE CONSTRUCTION
imputer = ColumnTransformer([
    ('impute_num', SimpleImputer(), slice(0, 180))
], remainder='passthrough')

standard_scaler = ColumnTransformer([
    # ('enc_nominal', OneHotEncoder(), slice(180, 200)),
    ('scale_num', StandardScaler(with_mean=False), slice(0, 180))
])

svc = SVC()

pipe = Pipeline([
    ('ohe', ohe),
    ('imputer', imputer),
    ('scaler', standard_scaler),
    ('svc', svc)
])

# GRID SEARCH
svc_params = {
    'svc__kernel': ['poly'],
    'svc__C': [0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 10],
    'svc__degree': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 'probability': [True, False]
}

t0 = time.time()
print('START GS')
gs = GridSearchCV(pipe, svc_params, cv=10, scoring='roc_auc', n_jobs=-1, verbose=10)
gs.fit(X_train, y_train)
gs_df = pd.DataFrame(gs.cv_results_)
print('END GS')
t1 = time.time()

print(f'time = {t1 - t0}')

y_pred = gs.predict(X_test)
