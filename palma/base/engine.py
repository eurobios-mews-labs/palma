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
from datetime import datetime
from typing import Dict

import pandas as pd
from flaml import AutoML
from sklearn import base

from palma.base.splitting_strategy import ValidationStrategy
from palma.utils.utils import get_hash


class BaseOptimizer(metaclass=ABCMeta):

    def __init__(self, engine_parameters: dict) -> None:
        self.__engine_parameters = engine_parameters
        self.__date = datetime.now()
        self.__run_id = get_hash(date=self.__date)
        self._problem = "unknown"

    @abstractmethod
    def optimize(self, X: pd.DataFrame, y: pd.Series,
                 splitter: "ValidationStrategy" = None
                 ) -> None:
        ...

    @property
    @abstractmethod
    def best_model_(self) -> None:
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

    def start(self, project: "Project"):
        from palma import logger
        self._problem = project.problem
        self.optimize(
            project.X.iloc[project.validation_strategy.train_index],
            project.y.iloc[project.validation_strategy.train_index],
            splitter=project.validation_strategy,
        )

        logger.logger.log_artifact(
            self.best_model_,
            self.__run_id)
        try:
            logger.logger.log_metrics(
                {"best_estimator": str(self.best_model_)}, 'optimizer'
            )
        except:
            pass

    @property
    def run_id(self) -> str:
        return self.__run_id

    @property
    def problem(self):
        return self._problem



class FlamlOptimizer(BaseOptimizer):
    def __init__(self, engine_parameters: dict) -> None:
        super().__init__(engine_parameters)

    def optimize(self, X: pd.DataFrame, y: pd.DataFrame,
                 splitter: ValidationStrategy = None
                 ) -> None:
        split_type = None if splitter is None else splitter.splitter
        groups = None if splitter is None else splitter.groups
        groups = groups if groups is None else groups[splitter.train_index]

        logging.disable()
        self.engine_parameters["task"] = self.problem
        self.allowing_splitter(splitter)
        self.__optimizer = AutoML()
        self.__optimizer.fit(
            X_train=pd.DataFrame(X.values, index=range(len(X))),
            y_train=pd.Series(y.values, index=range(len(X))),
            split_type=split_type, groups=groups,
            mlflow_logging=False,
            **self.engine_parameters
        )
        logging.basicConfig(level=logging.DEBUG)

    @property
    def best_model_(self) -> base.BaseEstimator:
        return self.__optimizer.model.model

    @property
    def transformer_(self):
        return self.__optimizer._transformer

    @property
    def allow_splitter(self):
        return True
