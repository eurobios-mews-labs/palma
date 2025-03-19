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
from palma import components
import matplotlib

__default_project_component__ = [
    # components.DeepCheck(),
]
__default_model_component__ = [
    components.ShapAnalysis(on='indexes_train_test', n_shap=200),]
__default_regression_component__ = [
    *__default_model_component__,
    components.RegressionAnalysis(on='indexes_train_test')
]
__default_scoring_component__ = [
    *__default_model_component__,
    components.ScoringAnalysis(on='indexes_train_test')
]


class AutoMl:
    """
    AutoMl - Automated Machine Learning

       Parameters
       ----------
       project_name : str
           Name of the machine learning project.
       problem : str
           Type of problem, either "classification" or "regression".
       X : pd.DataFrame
           Features of the training dataset.
       y : pd.Series
           Target variable of the training dataset.
       splitter
           Data splitter object for cross-validation.
       X_test : pd.DataFrame, optional
           Features of the test dataset, default is None.
       y_test : pd.Series, optional
           Target variable of the test dataset, default is None.
       groups : None, optional
           Grouping information for group-based cross-validation, default is None.

       Attributes
       ----------
       project : Project
           Machine learning project object.
       runner : ModelSelector
           Model selection and training engine.
       model : ModelEvaluation
           Model evaluation and analysis object.

       Methods
       -------
       run(engine_name, engine_parameter)
           Run the automated machine learning process using the specified engine.

       Notes
       -----
       The `AutoMl` class is designed to automate the machine learning pipeline,
       including project setup, model selection, and evaluation.

       Examples
       --------
       >>> automl = AutoMl(project_name='my-project',
       ...                 problem='classification',
       ...                 X=X,
       ...                 y=y,
       ...                 splitter=StratifiedKFold(n_splits=5))
       >>> automl.run(engine='FlamlEngine', engine_parameter={'time_budget': 20})
       """

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
        self.save_plt_backend = matplotlib.get_backend()
        matplotlib.use("agg")
        self.project = Project(project_name, problem)
        for c in __default_project_component__:
            self.project.add(c)
        self.project.start(X=X, y=y, splitter=splitter, X_test=X_test,
                           y_test=y_test,
                           groups=groups)
        matplotlib.use(self.save_plt_backend)

    def run(self, engine, engine_parameters):
        """
        Run the automated machine learning process.

        Parameters
        ----------
        engine : str
            Name of the engine to use.
        engine_parameters
            Parameters specific to the chosen machine learning engine.

        Returns
        -------
        self
        """
        from sklearn.base import clone
        self.save_plt_backend = matplotlib.get_backend()
        matplotlib.use("agg")
        self.runner = ModelSelector(engine, engine_parameters)
        self.runner.start(self.project)
        self.model = ModelEvaluation(self.runner.best_model_)
        if self.project.problem == "classification":
            [self.model.add(c) for c in __default_scoring_component__]
        elif self.project.problem == "regression":
            [self.model.add(c) for c in __default_regression_component__]
        self.model.fit(self.project)
        self.estimator_ = clone(self.runner.best_model_).fit(self.project.X, self.project.y)
        matplotlib.use(self.save_plt_backend)
        return self


