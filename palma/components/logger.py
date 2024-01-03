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

from palma.base.model_selection import ModelSelector
from palma.base.project import Project

try:
    import mlflow
except ImportError:
    pass

from palma.components.base import Logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


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
            logger.info(f"No {project.project_name} folder found,"
                        f" creating {self.path_study} folders")
            os.makedirs(self.path_study)

        artifact_name = f"{self.path_study}/project.pkl"

        with open(artifact_name, "wb") as output_file:
            logger.info(
                "Project instance saved in {}".format(artifact_name)
            )
            pickle.dump(project, output_file)
        self._log_params(
            {c.replace("_Project__", ""): str(v) for c, v in
             vars(project).items()}, "properties")

    def log_run(self, run: 'Run') -> None:
        """Deprecated"""
        pass

    def _log_metrics(self, metrics: dict, path: str) -> None:
        path = f"{self.path_study}/{path}.json"
        with open(path, 'w') as output_file:
            logger.info("Metrics saved in {}".format(path))
            json.dump(metrics, output_file, indent=4)

    def _log_model(self, estimator, path: str) -> None:
        path = f"{self.path_study}/{path}"
        with open(path, 'wb') as output_file:
            logger.info(f"Model saved in {path}")
            pickle.dump(estimator, output_file)

    def _log_params(self,
                    parameters: dict,
                    path: str) -> None:
        path = f"{self.path_study}/{path}.json"

        with open(path, 'w') as output_file:
            logger.info(f"Model's parameters saved in {path}")
            json.dump(parameters, output_file, indent=4)


class MLFlowLogger(Logger):
    def __init__(self, uri: str) -> None:
        super().__init__(uri)
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)

    def log_project(self, project: 'Project') -> None:
        print(project.project_name)
        mlflow.set_experiment(project.project_name)
        self.__project = project
        self.__project_path = os.path.join(tempfile.gettempdir(),
                                           project.project_name)
        artifact_name = os.path.join(self.__project_path,
                                     "project.pkl")
        if not os.path.exists(self.__project_path):
            os.makedirs(self.__project_path)
        with open(artifact_name, "wb") as output_file:
            print(
                "Project instance saved in {}".format(artifact_name)
            )
            pickle.dump(project, output_file)

    def log_run(self, run: 'ModelSelector') -> None:
        mlflow.set_experiment(self.__project.project_name)
        mlflow.start_run(run_name=run.run_id)
        self.__run_id_path = os.path.join(self.__project_path,
                                          run.run_id)
        if not os.path.exists(self.__project_path):
            os.makedirs(self.__project_path)
        if not os.path.exists(self.__run_id_path):
            print(
                f"Creating a temporary folder: {self.__run_id_path}"
            )
            os.mkdir(self.__run_id_path)

        self._log_artifacts({"model.pkl": run.engine.estimator_})
        self._log_params(run.engine.estimator_.get_params())

        self._log_artifacts({"project.pkl": self.__project})
        mlflow.set_tags({"project_id": self.__project.project_id})
        mlflow.end_run()

    def _log_metrics(self, metrics: dict[str, dict]) -> None:
        for k, v in metrics.items():
            mlflow.log_metrics(
                {k_ + "_" + k: v_ for k_, v_ in v.items()})

    def _log_artifacts(self, artifacts: dict) -> None:
        for l in artifacts.keys():
            path = os.path.join(self.__run_id_path, l)
            with open(path, 'wb') as file:
                pickle.dump(artifacts[l], file)

        mlflow.log_artifacts(self.__run_id_path)

    def _log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def _log_model(self):
        pass

    def log_learner(self, learner: "Learner"):
        mlflow.start_run(run_name=learner.id)
        self.__run_id_path = os.path.join(
            self.__project_path,
            learner.id)

        if not os.path.exists(self.__run_id_path):
            print(
                f"Creating a temporary folder: {self.__run_id_path}"
            )
            os.mkdir(self.__run_id_path)

        self._log_artifacts({"model.pkl": learner.__estimator})
        self._log_params(learner.__estimator.get_params())
        self._log_metrics(learner.metrics)
        self._log_artifacts({"project.pkl": self.__project})
        mlflow.set_tags({"project_id": self.__project.project_id})


def download(project_name: str, run_id: str, path: str, dst_path: str):
    """ Download an artifact file or directory from a run to a local directory if applicable,
        and return a local path for it.
    Parameters
    ----------
    run_id:
        The run to download artifacts from.
    path:
        Relative source path to the desired artifact.
    dst_path:
        Absolute path of the local filesystem destination directory to which to
        download the specified artifacts. This directory must already exist.
        If unspecified, the artifacts will either be downloaded to a new
        uniquely-named directory on the local filesystem or will be returned
        directly in the case of the LocalArtifactRepository.
    Returns
    -------
        Local path of desired artifact.
    """
    client = mlflow.tracking.MlflowClient()
    df = mlflow.search_runs(
        [mlflow.get_project_by_name(project_name).project_id], )
    id_mlflow = df.loc[df["tags.mlflow.runName"] == run_id, "run_id"].values[0]
    return client.download_artifacts(run_id=id_mlflow, path=path,
                                     dst_path=dst_path)


def load_pkl(project_name: str, run_id: str, path: str):
    """ Load an pickled artifact
    Parameters
    ----------
    project_name:
        Name of the use case
    run_id:
        The run to download artifacts from.
    path:
        Relative source path to the desired artifact.
    Returns
    -------
       python object.
    """
    dst_path = os.path.join(tempfile.gettempdir())
    output_file = download(project_name, run_id, path, dst_path)
    object_ = pickle.load(open(output_file, 'rb'))
    return object_
