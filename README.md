# Binary classification task

![Screenshot](/docs/readme-img.png)

Goal: Classify samples in `data/X_eval.npy`.

Given labeled data, find and train the best-performing classification model.

## Data

`X_public.npy`

- 600x200
- first 180 columns contain numeric values
- last 20 columns contain strings

`y_public.npy`

- 1x600
- labels for X_public

`X_eval.npy`

- 200x200
- column structure same as *X_public*

## Processes

`preprocessing.py` 

- Creates column transformer

`find_clf.py`

- Finds the best-performing classifier using RandomizedSearchCV

`find_params.py`

- Finds the best-performing parameters for the classifier found in `find_clf.py`

`predict_eval.py`

- Classifies the assigned samples

## Dependencies

- Python 3.9.7

- numpy 1.21.2

- pandas 1.3.4

- scikit-learn 1.0.1

- scipy 1.7.1

- All dependencies in `environment.yml`
