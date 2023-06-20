# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import pytest
from palma.components.checker import ValidationStrategyChecker
from sklearn import model_selection


def test_validation_checker(unbuilt_classification_project, regression_data):
    project = unbuilt_classification_project
    X = regression_data[0]
    y = regression_data[1]

    project.add(ValidationStrategyChecker())
    project.start(
        X, y,
        splitter=model_selection.ShuffleSplit(n_splits=4, random_state=42),
    )
