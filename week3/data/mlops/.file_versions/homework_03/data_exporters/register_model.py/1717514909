import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def register_model(df: pd.DataFrame,*args,**kwargs) -> None:
    
    MLFLOW_TRACKING_URI = "sqlite:///home/mlflow/mlflow.db"
    mlflow.set_experiment("LR-model")   
    mlflow.sklearn.autolog()
    
    with mlflow.start_run() as run:

        dv,lr = df
        #Save and log the artifact (dict vectorizer)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
       
        # Log the linear regression model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="sklearn-model",
            registered_model_name="linear-reg-model",
            )

