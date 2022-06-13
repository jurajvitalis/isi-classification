import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from old.data_preprocessing import X_train_final, X_test_final, y_train, y_test

clf = SVC(degree=2, kernel='poly', C=0.001, coef0=0.001, gamma=10, probability=True)
# clf.fit(X_train_final, y_train)
# y_pred = clf.predict(X_test_final)
# y_pred_prob = clf.predict_proba(X_test_final)

scores = cross_val_score(clf, X_train_final, y_train, cv=10, scoring="roc_auc", verbose=0)
print(f'CV score = {scores}')
print(f'CV mean = {np.mean(scores)}')

clf.fit(X_train_final, y_train)
y_pred = clf.predict(X_test_final)
y_pred_prob = clf.predict_proba(X_test_final)
print(f'SVC Accuracy = {roc_auc_score(y_test, y_pred)}')
print(f'SVC Accuracy = {roc_auc_score(y_test, y_pred_prob[:, 1])}')
print(f'Accuracy score = {accuracy_score(y_test, y_pred)}')
