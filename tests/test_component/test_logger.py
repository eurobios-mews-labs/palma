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

import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from palma import ModelEvaluation
from palma import ModelSelector
from palma import Project
from palma import logger, set_logger
from palma.components import FileSystemLogger, ScoringAnalysis
from palma.components import MLFlowLogger
from palma.components.logger import DummyLogger


def test_dummy_logger(classification_data):
    from palma import set_logger
    project = Project(problem="classification", project_name="test")
    path = tempfile.gettempdir() + "/mlflow"
    set_logger(DummyLogger(uri=path))
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(y)
    project.start(X, y, splitter=ShuffleSplit())
    ms = ModelSelector(engine="FlamlOptimizer",
                       engine_parameters=dict(time_budget=3))
    ms.start(project)

    model = ModelEvaluation(estimator=ms.best_model_)
    model.add(ScoringAnalysis(on="indexes_train_test"))
    model.fit(project)


def test_is_logged_project(build_classification_project):
    p = build_classification_project
    path_to_file = os.path.join(
        tempfile.gettempdir(),
        p.project_name,
        p.study_name,
        'project.pkl'
    )
    assert os.path.isfile(path_to_file), "project.pkl does not exist"


def test_uri_fs_logger():
    fsl = FileSystemLogger(uri=tempfile.gettempdir())
    assert fsl.uri == tempfile.gettempdir()


@pytest.fixture
def get_model_with_logger(build_classification_project):
    estimator = Pipeline(
        steps=[("scaler", StandardScaler()), ("est", LinearRegression())])
    model = ModelEvaluation(estimator)

    model.fit(build_classification_project)
    return model


def test_is_logged_project_existing_project_name(build_classification_project):
    path_to_file = os.path.join(
        tempfile.gettempdir(),
        build_classification_project.project_name,
        build_classification_project.study_name,
        'project.pkl'
    )
    assert os.path.isfile(path_to_file), "project.pkl does not exist"


def test_log_model(build_classification_project):
    engine_parameters = dict(time_budget=2, task='regression')
    ms = ModelSelector(
        engine='FlamlOptimizer',
        engine_parameters=engine_parameters,
    )
    ms.start(build_classification_project)


@pytest.fixture()
def get_mlflow_logger(classification_data):
    from palma import set_logger, logger

    set_logger(MLFlowLogger(uri=tempfile.gettempdir() + "/mlflow"))

    project = Project(problem="classification", project_name="test")
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(y)

    project.start(
        X,
        y,
        splitter=ShuffleSplit()
    )
    ms = ModelSelector(engine="FlamlOptimizer",
                       engine_parameters=dict(time_budget=3))
    ms.start(project)
    return logger


def test_is_logged_project_mlflow(get_mlflow_logger):
    get_mlflow_logger.logger.log_metrics({"a": 1})


def test_changing_logger():
    from palma import logger, set_logger

    set_logger(DummyLogger(uri="."))
    assert isinstance(logger.logger, DummyLogger)

    set_logger(MLFlowLogger(uri="."))
    assert isinstance(logger.logger, MLFlowLogger)
    set_logger(DummyLogger(uri="."))
    assert isinstance(logger.logger, DummyLogger)


def test_artifact_logging():
    set_logger(FileSystemLogger(uri="/tmp/mlflow"))

    fig = plt.figure()
    logger.logger.log_artifact('a', "text")
    logger.logger.log_metrics({'a': 1}, "metric")
    logger.logger.log_artifact(fig, "figure")

    set_logger(MLFlowLogger(uri="/tmp/mlflow/"))

    assert (logger.logger.uri == "/tmp/mlflow/")

    logger.logger.log_artifact('a', "text")
    logger.logger.log_metrics({'a': 1}, "metric")
    logger.logger.log_artifact(fig, "figure")
