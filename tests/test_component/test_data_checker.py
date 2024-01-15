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

import tempfile

import pandas as pd
from sklearn import model_selection

from palma import Project
from palma.components import FileSystemLogger
from palma.components.data_checker import DeepCheck, Leakage


def test_deep_check(classification_project):
    print(classification_project.X.columns)
    print(classification_project.y.name
          )
    dc = DeepCheck(dataset_parameters={"label": classification_project.y.name})
    dc(classification_project)


def test_leakage(classification_data):
    from palma import set_logger
    set_logger(FileSystemLogger(tempfile.gettempdir()))
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(y)
    project = Project(problem="classification", project_name="test")
    project.add(Leakage())
    project.start(
        X, y,
        splitter=model_selection.ShuffleSplit(n_splits=4, random_state=42))

