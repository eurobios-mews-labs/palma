# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from datetime import datetime
from hashlib import blake2b
from typing import List

import pandas as pd

from palma.base.splitting_strategy import ValidationStrategy
from palma.utils.names import get_random_name
from palma.utils.utils import check_started


class Project(object):

    def __init__(
            self,
            project_name: str,
            problem: str
    ) -> None:

        self.__project_name = project_name
        self.__study_name = get_random_name()
        self.__date = datetime.now()
        self.__problem = problem

        self.__components = {}

        self.__is_started = False
        self.__component_list = []

    @check_started("You cannot add a Component for a started Project")
    def add(self, component: "Component") -> None:
        from palma.components.base import ProjectComponent, Logger

        self.__component_list.append(str)
        if isinstance(component, ProjectComponent):
            self.__components.update({str(component): component})
        elif isinstance(component, Logger) or hasattr(component, "log_project"):
            self._logger = component

        else:
            raise TypeError(
                "The added component must be an instance of class Component"
            )

    @check_started("You cannot restart an Project")
    def start(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            splitter,
            X_test=None, y_test=None,
            groups=None,
            **kwargs
    ) -> None:
        from palma.utils.checker import ProjectPlanChecker
        self.__validation_strategy = ValidationStrategy(splitter)
        self.__base_index = list(range(len(X)))
        self.__X, self.__y = self.__validation_strategy(
            X=X, y=y, X_test=X_test, y_test=y_test,
            groups=groups)
        ProjectPlanChecker().run_checks(self)
        self.__data_id = blake2b(
            pd.util.hash_pandas_object(
                pd.concat([self.__X, self.__y], axis=1)
            ).values, digest_size=5
        ).hexdigest()

        self.__call_components(self)

        self.__is_started = True

        if hasattr(self, "_logger"):
            self._logger.log_project(self)

    def __call_components(self, object_: "Project") -> None:
        for _, component in self.components.items():
            component(object_)

    @property
    def base_index(self) -> List[int]:
        return self.__base_index

    @property
    def components(self) -> dict:
        return self.__components

    @property
    def date(self) -> datetime:
        return self.__date

    @property
    def project_id(self) -> str:
        return f"{self.__data_id}_{self.validation_strategy.id}"

    @property
    def is_started(self) -> bool:
        return self.__is_started

    @property
    def problem(self) -> str:
        return self.__problem

    @property
    def validation_strategy(self) -> 'ValidationStrategy':
        return self.__validation_strategy

    @property
    def project_name(self) -> str:
        return self.__project_name

    @property
    def study_name(self) -> str:
        return self.__study_name

    @property
    def X(self) -> pd.DataFrame:
        return self.__X.copy()

    @property
    def y(self) -> pd.Series:
        return self.__y.copy()
