# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import pickle
import tempfile
import typing
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt

from palma.base.project import Project

try:
    import mlflow
except ImportError:
    mlflow = None

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.NOTSET)


class Logger(metaclass=ABCMeta):
    """
    Logger is an abstract class that defines a common
    interface for a set of Logger-subclasses.

    It provides common methods for all possible subclasses, making it
    possible for a user to create a custom subclass compatible  with
    the rest of the components.
    """

    def __init__(self, uri: str, **kwargs) -> None:
        self.__uri = uri

    @abstractmethod
    def log_project(self, project: 'Project') -> None:
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict, path: str) -> None:
        ...

    @abstractmethod
    def log_params(self, **kwargs) -> None:
        ...

    @abstractmethod
    def log_artifact(self, **kwargs) -> None:
        ...

    @property
    def uri(self):
        return self.__uri


class DummyLogger(Logger):

    def log_project(self, project: 'Project') -> None:
        pass

    def log_metrics(self, metrics: dict, path: str) -> None:
        pass

    def log_params(self, parameters: dict,
                   path: str) -> None:
        pass

    def log_artifact(self, obj, path: str) -> None:
        pass


class FileSystemLogger(Logger):
    """

    Parameters
    ----------
    uri : str
        root path or directory, from which will be saved artifacts and metadata 
    """

    def __init__(self, uri: str = tempfile.gettempdir(), **kwargs) -> None:
        super().__init__(uri, **kwargs)
        self.path_project = f"{self.uri}/unknown_project"
        self.path_study = f"{self.path_project}/unknown_run"

    def log_project(self, project: 'Project') -> None:
        """
        log_project performs the first level of backup as described
        in the object description. 

        This method creates the needed folders and saves an instance of \
        :class:`~palma.Project`.

        Parameters
        ----------
        project: :class:`~palma.Project`
            an instance of Project
        """
        self.path_project = f"{self.uri}/{project.project_name}"
        self.path_study = f"{self.path_project}/{project.study_name}"

        self.create_directories()

        artifact_name = f"{self.path_study}/project.pkl"

        with open(artifact_name, "wb") as output_file:
            _logger.info(
                "Project instance saved in {}".format(artifact_name)
            )
            pickle.dump(project, output_file)
        self.log_params(
            {c.replace("_Project__", ""): str(v) for c, v in
             vars(project).items()}, "properties")

    def log_metrics(self, metrics: dict, path: str) -> None:
        path = f"{self.path_study}/{path}.json"
        with open(path, 'w') as output_file:
            _logger.info("Metrics saved in {}".format(path))
            json.dump(metrics, output_file, indent=4)

    def log_artifact(self, obj, path: str) -> None:
        path = f"{self.path_study}/{path}"
        self.create_directories()
        with open(path, 'wb') as output_file:
            if isinstance(obj, plt.Figure):
                obj.savefig(f"{path}.png")
            else:
                pickle.dump(obj, output_file)

    def log_params(self,
                   parameters: dict,
                   path: str) -> None:
        path = f"{self.path_study}/{path}.json"

        with open(path, 'w') as output_file:
            _logger.info(f"Model's parameters saved in {path}")
            json.dump(parameters, output_file, indent=4)

    def create_directories(self):
        if not os.path.exists(self.path_study):
            _logger.info(f"No {self.path_study} folder found,"
                         f" creating {self.path_study} folders")
            os.makedirs(self.path_study)


class MLFlowLogger(Logger):
    """
    MLFlowLogger class for logging experiments using MLflow.

    Parameters
    ----------
    - uri (str): The URI for the MLflow tracking server.

    Attributes
    ----------
    - tmp_logger (FileSystemLogger): Temporary logger for local logging
    before MLflow logging.

    Methods
    -------

    log_project(project: 'Project') -> None:
        Logs the project information to MLflow, including project name and parameters.

    log_metrics(metrics: dict[str, typing.Any]) -> None:
        Logs metrics to MLflow.

    log_artifact(artifact: dict, path) -> None:
        Logs artifacts to MLflow using the temporary logger.

    log_params(params: dict) -> None:
        Logs parameters to MLflow.

    log_model(model, path) -> None:
        Logs the model to MLflow using the temporary logger.

    Raises
    ------
    ImportError: If mlflow is not installed.

    Example
    -------
    ```python
    mlflow_logger = MLFlowLogger(uri="http://mlflow-server:5000")
    mlflow_logger.log_project(my_project)
    mlflow_logger.log_metrics({"accuracy": 0.95, "loss": 0.05})
    mlflow_logger.log_artifact(my_artifact, "path/to/artifact")
    mlflow_logger.log_params({"param1": 42, "param2": "value"})
    mlflow_logger.log_model(my_model, "path/to/model")
    ```

    Note
    ----

    Ensure that MLflow is installed before using this class.

    """

    def __init__(self, uri: str) -> None:
        if mlflow is None:
            raise ImportError("mlflow is not installed")
        super().__init__(uri)
        mlflow.set_tracking_uri(uri)
        self.file_system_logger = FileSystemLogger()

    def log_project(self, project: 'Project') -> None:
        mlflow.set_experiment(
            project.project_name)
        self.file_system_logger.log_project(project)
        self.log_params({c.replace("_Project__", ""): str(v) for c, v in
                         vars(project).items()})

    def log_metrics(self, metrics: dict[str, typing.Any], path=None) -> None:
        mlflow.log_metrics(
            {k: v for k, v in metrics.items()})

    def log_artifact(self, artifact: dict, path) -> None:
        self.file_system_logger.log_artifact(artifact, path)

        mlflow.log_artifacts(f"{self.file_system_logger.path_study}")

    def log_params(self, params: dict) -> None:
        mlflow.log_params({k: str(v)[:100] for k, v in params.items()})

    def log_model(self, model, path):
        self.file_system_logger.log_artifact(model, path)
        mlflow.log_artifacts(f"{self.file_system_logger.path_study}")


class _Logger:
    def __init__(self, dummy) -> None:
        self.__logger = dummy

    def __set__(self, logger) -> None:
        """
        
        Parameters
        ----------
        logger: Logger
            Define the logger to use.

            >>> from palma import logger, set_logger
            >>> from palma.components import FileSystemLogger
            >>> from palma.components import MLFlowLogger
            >>> set_logger(MLFlowLogger(uri="."))
            >>> set_logger(FileSystemLogger(uri="."))
        Returns
        -------
            None
        """
        self.__logger = logger

    @property
    def logger(self) -> Logger:
        return self.__logger


logger = _Logger(DummyLogger("."))


set_logger = logger.__set__