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

from datetime import datetime
from hashlib import blake2b

import pandas as pd

from palma.base.splitting_strategy import ValidationStrategy
from palma.utils.utils import check_started


class Project:
    """
    Represents a machine learning project with various components
    and logging capabilities.

    Parameters
    ----------
    project_name (str): The name of the project.
    problem (str): The description of the machine learning problem.
                   Accepted values: "classification" or "regression".

    Attributes
    ----------
    base_index (List[int]): List of base indices for the project.
    components (dict): Dictionary containing project components.
    date (datetime): The date and time when the project was created.
    project_id (str): Unique identifier for the project.
    is_started (bool): Indicates whether the project has been started.
    problem (str): Description of the machine learning problem.
    validation_strategy (ValidationStrategy): The validation strategy used in the project.
    project_name (str): The name of the project.
    study_name (str): The randomly generated study name.
    X (pd.DataFrame): The feature data for the project.
    y (pd.Series): The target variable for the project.

    Methods
    -------
    add(component: Component) -> None: Adds a component to the project.
    start(X: pd.DataFrame, y: pd.Series, splitter, X_test=None, y_test=None, groups=None, **kwargs) -> None:
        Starts the project with the specified data and validation strategy.
    """

    def __init__(
            self,
            project_name: str,
            problem: str
    ) -> None:

        self.__project_name = project_name
        self.__date = datetime.now()
        self.__study_name = self.date
        self.__problem = problem

        self.__components = {}
        self.__is_started = False
        self.__component_list = []

    @check_started("You cannot add a Component for a started Project")
    def add(self, component: "Component") -> None:
        from palma.components.base import ProjectComponent
        self.__component_list.append(str)
        if isinstance(component, ProjectComponent):
            self.__components.update({str(component): component})
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
        from palma import logger
        self.__validation_strategy = ValidationStrategy(splitter)
        self.__X, self.__y = self.__validation_strategy(
            X=X, y=y, X_test=X_test, y_test=y_test,
            groups=groups)
        ProjectPlanChecker().run_checks(self)
        self.__data_id = blake2b(
            pd.util.hash_pandas_object(
                pd.concat([self.__X, self.__y], axis=1)
            ).values, digest_size=5
        ).hexdigest()
        logger.logger.log_project(self)
        self.__call_components(self)
        self.__is_started = True

    def __call_components(self, object_: "Project") -> None:
        for _, component in self.components.items():
            component(object_)

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
        return str(self.date)

    @property
    def X(self) -> pd.DataFrame:
        return self.__X.copy()

    @property
    def y(self) -> pd.Series:
        return self.__y.copy()
