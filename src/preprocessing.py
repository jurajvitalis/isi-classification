from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Create transformers
ohe = OneHotEncoder(handle_unknown='ignore')
imputer = SimpleImputer()
scaler = StandardScaler()

# Set up pipeline for numerical columns
numeric_pipe = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler)
])

# Combine pipelines into a single transformer
col_transformer = ColumnTransformer([
    ('numeric', numeric_pipe, slice(0, 180)),
    ('categorical', ohe, slice(180, 200))
])
