import os
from dotenv import load_dotenv
import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the environment variables
load_dotenv()

# Set env variable
APP_URI = os.getenv("$APP_URI")
EXPERIMENT_NAME = "mlflow-sklearn-iris"

# Load Iris dataset
iris = load_iris()

# Split dataset into X features and Target variable
X = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
y = pd.Series(data=iris["target"], name="target")

# Split our training set and our test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Visualize dataset
X_train.head()

# Set tracking URI to your Heroku application
mlflow.set_tracking_uri(APP_URI)

# Set experiment's info
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Call mlflow autolog to log everything by default
mlflow.sklearn.autolog()

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Specified Parameters
    c_param = 0.5

    # Instanciate and fit the model
    lr = LogisticRegression(C=c_param)
    lr.fit(X_train.values, y_train.values)

    # Store metrics
    predicted_qualities = lr.predict(X_test.values)
    accuracy = lr.score(X_test.values, y_test.values)

    # Print results
    print("LogisticRegression model")
    print("Accuracy: {}".format(accuracy))
