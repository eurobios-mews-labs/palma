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

import pandas as pd

from palma import Project, ModelSelector, ModelEvaluation
from palma.components import FileSystemLogger


class AutoMl:

    def __init__(self,
                 project_name: str,
                 problem: str,
                 X: pd.DataFrame,
                 y: pd.Series,
                 splitter,
                 X_test=None,
                 y_test=None,
                 groups=None,

                 ):
        self.project = Project(project_name, problem)
        self.project.add(FileSystemLogger())
        self.project.start(X=X, y=y, splitter=splitter, X_test=X_test,
                           y_test=y_test,
                           groups=groups)

    def run(self, engine_name, engine_parameter):
        self.runner = ModelSelector(engine_name, engine_parameter)
        self.runner.start(self.project)
        self.model = ModelEvaluation(self.runner.best_model_)
        self.model.add()
