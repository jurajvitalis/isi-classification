import joblib
import numpy as np


# Load data
X_eval = np.load('../data/X_eval400.npy', allow_pickle=True)

# Load pipeline object from gridsearch
gs = joblib.load('gs_result.pkl')

best_pipeline = gs.best_estimator_
y_pred = best_pipeline.predict(X_eval)

np.save('y_predikcia', y_pred, allow_pickle=True)
