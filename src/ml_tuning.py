import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

def test_models(models, data, target_name, numeric_features = [], categorical_features = [], test_size = 0.2, iterations = 50):
    """
    Return a dict with r2 score for train and test for a list of models
    Example of results value:
    {
      'Ridge': { 'r2_train': 0.92, 'r2_test': 0.88 },
      'Lasso': { 'r2_train': 0.92, 'r2_test': 0.88 },
      ...
    }

    Arguments:
    """

    results = {}

    transformers = []

    # Create pipeline for numeric features
    if (len(numeric_features) > 0):
      numeric_transformer = Pipeline(
          steps=[
              ('imputer', SimpleImputer(strategy='median')), # missing values will be replaced by columns' mean
              ('scaler', StandardScaler())
          ]
      )
      transformers.append(('num', numeric_transformer, numeric_features))

    # Create pipeline for categorical features
    if (len(categorical_features) > 0):
      categorical_transformer = Pipeline(
          steps=[
              ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')) # first column will be dropped to avoid creating correlations between features
          ]
      )
      transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers = transformers
    )

    y = data.loc[:,target_name]
    X = data.drop(target_name, axis = 1)
    
    for model_key in models:
        r2_train = []
        r2_test = []

        for j in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)

            # Preprocessings on train set
            X_train = preprocessor.fit_transform(X_train)

            # Preprocessings on test set
            X_test = preprocessor.transform(X_test)

            # Train model
            regressor = models[model_key]
            regressor.fit(X_train, y_train)

            # Predictions on training set
            y_train_pred = regressor.predict(X_train)

            # Predictions on test set
            y_test_pred = regressor.predict(X_test)

            r2_train.append(r2_score(y_train, y_train_pred))
            r2_test.append(r2_score(y_test, y_test_pred))

        # Add model mean r2 score for train and test 
        results[model_key] = {'r2_train': np.mean(r2_train), 'r2_test': np.mean(r2_test)}
        
    return pd.DataFrame(results).transpose()
