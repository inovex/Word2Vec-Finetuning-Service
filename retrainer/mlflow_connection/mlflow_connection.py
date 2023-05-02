"""Log models to mlflow."""

import os
import mlflow
from mlflow.tracking import MlflowClient
from retrainer.mlflow_connection.exceptions import *


class MlflowConnection:
    """Organize MlFlow Model stages."""

    def __init__(self, experiment_name):
        mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI"))
        self.client = MlflowClient()
        # Use the config file here once it exists
        self.experiment_name = experiment_name
        self.model_name = experiment_name + "_model"
        if not mlflow.get_experiment_by_name(self.experiment_name):
            raise MlflowExperimentNotFoundError(self.experiment_name)

    def init_mlflow(self):
        if not mlflow.get_experiment_by_name(self.experiment_name):
            mlflow.create_experiment(name=self.experiment_name)
        experiment = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        return experiment
