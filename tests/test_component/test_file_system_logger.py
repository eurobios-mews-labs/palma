# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import os
import tempfile

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from palma import ModelEvaluation
from palma import Project
from palma.components import FileSystemLogger


@pytest.fixture
def build_classification_project(unbuilt_classification_project,
                                 classification_data):
    project = Project(problem="classification", project_name="test")
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(y)
    project.add(FileSystemLogger(tempfile.gettempdir()))
    project.start(
        X,
        y,
        splitter=ShuffleSplit()
    )
    return project


def test_is_logged_project(build_classification_project):
    path_to_file = os.path.join(
        tempfile.gettempdir(),
        build_classification_project.project_name,
        build_classification_project.project_name,
        'project.pkl'
    )
    assert os.path.isfile(path_to_file), "project.pkl does not exist"


@pytest.fixture
def get_model_with_logger(build_classification_project):
    estimator = Pipeline(
        steps=[("scaler", StandardScaler()), ("est", LinearRegression())])
    model = ModelEvaluation(estimator)

    model.fit(build_classification_project)
    return model


def test_is_logged_params_flaml(get_model_with_logger, build_classification_project):

    path_to_hp_parameters = os.path.join(
        tempfile.gettempdir(),
        build_classification_project.project_name,
        build_classification_project.project_name,
        get_model_with_logger.id,
        'run_parameters.json'
    )
    path_to_models_parameters = os.path.join(
        tempfile.gettempdir(),
        build_classification_project.project_name,
        build_classification_project.project_name,
        get_model_with_logger.id,
        'model_parameters.json'
    )
    assert os.path.isfile(path_to_hp_parameters) and \
        os.path.isfile(path_to_models_parameters), "run_parameters.json and \
            model_parameters.json does not exist"


def test_is_logged_metrics_flaml(get_model_with_logger, build_classification_project):
    path_to_metrics = os.path.join(
        tempfile.gettempdir(),
        build_classification_project.project_name,
        build_classification_project.project_name,
        get_model_with_logger.id,
        'metrics.json'
    )
    assert os.path.isfile(path_to_metrics), "metrics.json does not exist"


def test_is_logged_project_existing_project_name(get_model_with_logger, build_classification_project):

    path_to_file = os.path.join(
        tempfile.gettempdir(),
        build_classification_project.project_name,
        build_classification_project.project_name,
        'project.pkl'
    )
    assert os.path.isfile(path_to_file), "project.pkl does not exist"
