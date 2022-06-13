# Binary classification task

Given labeled data, find and train the best classification model.

Goal: Classify samples in *X_eval*.

## Data

*X_public*

- 600x200
- first 180 columns contain numeric values
- last 20 columns contain strings

*y_public*

- 1x600
- labels for X_public

*X_eval*

- 200x200
- column structure same as *X_public*

## Files

`preprocessing.py` 

- Creates column transformer

`find_clf.py`

- Finds the optimal classifier using RandomizedSearchCV

`find_params.py`

- Finds the optimal parameters for the classifier found in `find_clf.py`

`predict_eval.py`

- Classifies the assigned samples

## TODO

- [x] Create pipelines for everything
