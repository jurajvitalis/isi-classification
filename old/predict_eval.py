# Training the classifier on X_public
# Predicting X_eval

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

X_pub = np.load('../data/X_public400.npy', allow_pickle=True)
y_pub = np.load('../data/y_public400.npy', allow_pickle=True)
X_eval = np.load('../data/X_eval400.npy', allow_pickle=True)

# Split dataset by data type
X_pub_float = X_pub[:, :180]
X_pub_str = X_pub[:, 180:]

X_eval_float = X_eval[:, :180]
X_eval_str = X_eval[:, 180:]

# Use OHE on string data
ohe = OneHotEncoder(handle_unknown='ignore')
X_pub_str_ohe = ohe.fit_transform(X_pub_str)
X_eval_str_ohe = ohe.transform(X_eval_str)

# Use IMPUTER on float data
imputer = SimpleImputer()
X_pub_float_imp = imputer.fit_transform(X_pub_float)
X_eval_float_imp = imputer.transform(X_eval_float)

# Use SCALER on float data
scaler = StandardScaler()
X_pub_float_imp_scaled = scaler.fit_transform(X_pub_float_imp)
X_eval_float_imp_scaled = scaler.transform(X_eval_float_imp)

# Combine float and string data
X_pub_final = np.append(X_pub_float_imp_scaled, X_pub_str_ohe.toarray(), axis=1)
X_eval_final = np.append(X_eval_float_imp_scaled, X_eval_str_ohe.toarray(), axis=1)

# Train the model
svc = SVC(degree=2, kernel='poly', coef0=0, C=1)
svc.fit(X_pub_final, y_pub)
y_pred = svc.predict(X_eval_final)

np.save("y_predikcia", y_pred, allow_pickle=True)
