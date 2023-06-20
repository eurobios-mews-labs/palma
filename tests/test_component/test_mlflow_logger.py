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

import pytest
from sklearn.model_selection import ShuffleSplit

from palma.base.project import Project
from palma.components.logger import MLFlowLogger

try:
    print(os.environ['MLFLOW_TRACKING_URI'])

    @pytest.fixture
    def build_classification_project(get_classification_data):
        project = Project(
            project_name='__test_logger__',
            problem='classification'
        )
        project.add(MLFlowLogger(os.environ['MLFLOW_TRACKING_URI']))
        project.start(
            get_classification_data.X,
            get_classification_data.y,
            metrics_to_track=['recall', 'precision'],
            splitter=ShuffleSplit(n_splits=3)
        )
        return project

except ModuleNotFoundError:
    pass
