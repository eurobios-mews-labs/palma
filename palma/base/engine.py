# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Union

import pandas as pd
from flaml import AutoML
from sklearn import base

from palma.base.splitting_strategy import ValidationStrategy

try:
    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor
except ImportError:
    pass


class BaseOptimizer(metaclass=ABCMeta):

    def __init__(self, engine_parameters: dict) -> None:
        self.__engine_parameters = engine_parameters

    @abstractmethod
    def optimize(self, X: pd.DataFrame, y: pd.Series,
                 splitter: "ValidationStrategy" = None
                 ) -> None:
        ...

    @property
    @abstractmethod
    def optimizer(self) -> None:
        ...

    @property
    @abstractmethod
    def estimator_(self) -> None:
        ...

    @property
    @abstractmethod
    def transformer_(self) -> None:
        ...

    @property
    def engine_parameters(self) -> Dict:
        return self.__engine_parameters

    @property
    def allow_splitter(self):
        return False

    def allowing_splitter(self, splitter):
        if not self.allow_splitter and splitter is not None:
            logging.warning(f"Optimizer does not support splitter {splitter}")


class AutoSklearnOptimizer(BaseOptimizer):

    def __init__(self, problem: str, engine_parameters: dict) -> None:
        super().__init__(engine_parameters)
        self.problem = problem

    def optimize(self, X: pd.DataFrame, y: pd.Series, splitter=None) -> None:
        if self.problem == "classification":
            self.__optimizer = self.AutoSklearnClassifier(
                **self.engine_parameters
            )
        elif self.problem == "regression":
            self.__optimizer = self.AutoSklearnRegressor(
                **self.engine_parameters
            )
        else:
            raise ValueError(
                f"{self.problem} problem not compatible with autosklearn engine"
            )

        self.__optimizer.fit(X, y)
        self.__optimizer.refit(X, y)

    @property
    def optimizer(self) -> Union[
        'AutoSklearnClassifier',
        'AutoSklearnRegressor']:
        return self.__optimizer

    @property
    def estimator_(self) -> Union[
        'AutoSklearnClassifier',
        'AutoSklearnRegressor']:
        return self.__optimizer.get_models_with_weights()

    @property
    def transformer_(self):
        ...


class FlamlOptimizer(BaseOptimizer):
    def __init__(self, problem: str, engine_parameters: dict) -> None:
        super().__init__(engine_parameters)

        if "task" in engine_parameters.keys():
            logging.info(
                f"The problem is already provided through project object ({problem})"
            )
        engine_parameters["task"] = problem

    def optimize(self, X: pd.DataFrame, y: pd.DataFrame,
                 splitter: ValidationStrategy = None
                 ) -> None:
        split_type = None if splitter is None else splitter.splitter
        groups = None if splitter is None else splitter.groups
        groups = groups if groups is None else groups[splitter.train_index]

        self.allowing_splitter(splitter)
        self.__optimizer = AutoML()
        self.__optimizer.fit(
            X_train=pd.DataFrame(X.values, index=range(len(X))),
            y_train=pd.Series(y.values, index=range(len(X))),
            split_type=split_type, groups=groups,
            mlflow_logging=False,
            **self.engine_parameters
        )

    @property
    def optimizer(self) -> AutoML:
        return self.__optimizer

    @property
    def estimator_(self) -> 'base.BaseEstimator':
        return self.__optimizer.model.model

    @property
    def transformer_(self) -> 'flaml.data.DataTransformer':
        return self.__optimizer._transformer

    @property
    def allow_splitter(self):
        return True
