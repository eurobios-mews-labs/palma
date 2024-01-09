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

from palma.base.project import Project

try:
    import mlflow
except ImportError:
    pass

from palma.components.base import Logger

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.NOTSET)


class DummyLogger(Logger):

    def log_project(self, project: 'Project') -> None:
        pass

    def _log_metrics(self, **kwargs) -> None:
        pass

    def _log_params(self, **kwargs) -> None:
        pass

    def _log_model(self, **kwargs) -> None:
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

        if not os.path.exists(self.path_study):
            _logger.info(f"No {project.project_name} folder found,"
                        f" creating {self.path_study} folders")
            os.makedirs(self.path_study)

        artifact_name = f"{self.path_study}/project.pkl"

        with open(artifact_name, "wb") as output_file:
            _logger.info(
                "Project instance saved in {}".format(artifact_name)
            )
            pickle.dump(project, output_file)
        self._log_params(
            {c.replace("_Project__", ""): str(v) for c, v in
             vars(project).items()}, "properties")

    def _log_metrics(self, metrics: dict, path: str) -> None:
        path = f"{self.path_study}/{path}.json"
        with open(path, 'w') as output_file:
            logger.info("Metrics saved in {}".format(path))
            json.dump(metrics, output_file, indent=4)

    def _log_model(self, estimator, path: str) -> None:
        path = f"{self.path_study}/{path}"
        with open(path, 'wb') as output_file:
            _logger.info(f"Model saved in {path}")
            pickle.dump(estimator, output_file)

    def _log_params(self,
                    parameters: dict,
                    path: str) -> None:
        path = f"{self.path_study}/{path}.json"

        with open(path, 'w') as output_file:
            _logger.info(f"Model's parameters saved in {path}")
            json.dump(parameters, output_file, indent=4)


class MLFlowLogger(Logger):
    def __init__(self, uri: str) -> None:
        super().__init__(uri)
        mlflow.set_tracking_uri(uri)
        self.tmp_logger = FileSystemLogger()

    def log_project(self, project: 'Project') -> None:
        mlflow.set_experiment(
            project.project_name)
        self.tmp_logger.log_project(project)
        self._log_params(
            {c.replace("_Project__", ""): str(v) for c, v in
             vars(project).items()})

    def _log_metrics(self, metrics: dict[str, typing.Any]) -> None:
        mlflow.log_metrics(
            {k: v for k, v in metrics.items()})

    def _log_artifact(self, artifact: dict, path) -> None:
        self.tmp_logger._log_model(artifact, path)

        mlflow.log_artifacts(f"{self.tmp_logger.path_study}/{path}")

    def _log_params(self, params: dict) -> None:
        mlflow.log_params({k: str(v)[:100] for k, v in params.items()})

    def _log_model(self, model, path):
        self.tmp_logger._log_model(model, path)
        mlflow.log_artifacts(f"{self.tmp_logger.path_study}")


class Log:
    def __init__(self, dummy):
        self.logger = dummy

    def __set__(self, logger):
        self.logger = logger


logger = Log(DummyLogger)


def set(log_object):
    logger.__set__(log_object)
