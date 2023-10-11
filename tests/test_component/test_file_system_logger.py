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

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from palma import ModelEvaluation


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


def test_is_logged_project_existing_project_name(get_model_with_logger,
                                                 build_classification_project):
    path_to_file = os.path.join(
        tempfile.gettempdir(),
        build_classification_project.project_name,
        build_classification_project.project_name,
        'project.pkl'
    )
    assert os.path.isfile(path_to_file), "project.pkl does not exist"
