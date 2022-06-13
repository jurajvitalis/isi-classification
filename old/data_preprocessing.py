import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer


X_eval = np.load('../data/X_eval400.npy', allow_pickle=True)
X_public = np.load('../data/X_public400.npy', allow_pickle=True)
y_public = np.load('../data/y_public400.npy', allow_pickle=True)

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
# Add OHE dummies
X_public = np.hstack((X_public, ohe_dummies[:, 0:180]))

# Split the public dataset
X_train, X_test, y_train, y_test = train_test_split(X_public, y_public,
                                                    test_size=0.2, random_state=1, stratify=y_public)

# Transform the training set
# Impute NaN values from the training set
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train)

# Scale float values from the training set
scaler = StandardScaler()
X_train_imputed_scaled = scaler.fit_transform(X_train_imputed[:, 0:180])
X_train_final = np.hstack((X_train_imputed_scaled, X_train_imputed[:, 180:360]))

# Transform the testing set
X_test_imputed = imputer.transform(X_test)
X_test_imputed_scaled = scaler.transform(X_test_imputed[:, 0:180])
X_test_final = np.hstack((X_test_imputed_scaled, X_test_imputed[:, 180:360]))

# Transform X_eval
# OHE
X_eval_nominal = X_eval[:, 180:201]
X_eval_ohe_dummies = ohe.transform(X_eval_nominal).toarray()
X_eval = np.delete(X_eval, np.s_[180:201], axis=1)
X_eval = np.hstack((X_eval, X_eval_ohe_dummies[:, 0:180]))

# Impute
X_eval_imputed = imputer.transform(X_eval)

# Scale
X_eval_imputed_scaled = scaler.transform(X_eval_imputed[:, 0:180])

# Add OHE dummies
X_eval_final = np.hstack((X_eval_imputed_scaled, X_eval_imputed[:, 180:360]))
