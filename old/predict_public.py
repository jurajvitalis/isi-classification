# Training the classifier on the training set of X_public
# Predicting the testing set of X_public

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statistics

X_public = np.load('../data/X_public400.npy', allow_pickle=True)
y_public = np.load('../data/y_public400.npy', allow_pickle=True)

roc_auc_score_list = []
cv_score_list = []

for i in range(0, 100):

    # Split public data into train/test, 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_public, y_public, test_size=0.20, stratify=y_public, random_state=i)

    # Transform the training set

    # Split train data by data type
    X_train_float = X_train[:, :180]
    X_train_str = X_train[:, 180:]

    # Split test data by data type
    X_test_float = X_test[:, :180]
    X_test_str = X_test[:, 180:]

    # Use OHE on string data
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_train_str_ohe = ohe.fit_transform(X_train_str)
    X_test_str_ohe = ohe.transform(X_test_str)

    # Use IMPUTER on float data
    imp = SimpleImputer()
    X_train_float_imp = imp.fit_transform(X_train_float)
    X_test_float_imp = imp.transform(X_test_float)

    # Use SCALER on float data
    scaler = StandardScaler()
    X_train_float_imp_scaled = scaler.fit_transform(X_train_float_imp)
    X_test_float_imp_scaled = scaler.transform(X_test_float_imp)

    # Combine float and string data
    X_train_final = np.append(X_train_float_imp_scaled, X_train_str_ohe.toarray(), axis=1)
    X_test_final = np.append(X_test_float_imp_scaled, X_test_str_ohe.toarray(), axis=1)

    # Train an SVC model
    svc = SVC(degree=2, kernel='poly', coef0=0, C=1)
    svc.fit(X_train_final, y_train)

    # Evaluate the result
    print('random_state = ', i)

    y_pred = svc.predict(X_test_final)

    # Roc_auc_score
    roc_auc_score = metrics.roc_auc_score(y_test, y_pred)
    roc_auc_score_list.append(roc_auc_score)
    print("roc auc score: ", roc_auc_score, "s ", svc)

    # With cross validation
    cv_score = cross_val_score(svc, X_test_final, y_test, scoring='roc_auc', cv=10)
    cv_score_list.append(cv_score.mean())
    print("CV roc_auc: ", cv_score.mean(), "s ", svc)
    print()

print(f'Average roc_auc_score = {statistics.mean(roc_auc_score_list)}')
print(f'Average cv_score = {statistics.mean(cv_score_list)}')




