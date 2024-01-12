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

import logging
from typing import List, Union

import pandas as pd
from deepchecks.core import BaseCheck, BaseSuite
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.suites.default_suites import (train_test_validation,
                                                      data_integrity)

from palma.base.project import Project
from palma.components.base import ProjectComponent
from palma.components.logger import logger
from palma.utils import utils


class DeepCheck(ProjectComponent):
    """
    This object is a wrapper of the Deepchecks library and allows to audit the
    data through various checks such as data drift, duplicate values, ...

    Parameters
    ----------
    dataset_parameters : dict, optional
        Parameters and their values that will be used to generate
        :class:`deepchecks.Dataset` instances (required to run the checks on)
    dataset_checks: Union[List[BaseCheck], BaseSuite], optional
        List of checks or suite of checks that will be run on the whole dataset
        By default: use the default suite single_dataset_integrity to detect
        the integrity issues
    train_test_datasets_checks: Union[List[BaseCheck], BaseSuite], optional
        List of checks or suite of checks to detect issues related to the
        train-test split, such as feature drift, detecting data leakage...
        By default: use the default suites train_test_validation and
        train_test_leakage
    """

    def __init__(
            self,
            name: str = 'Data Checker',
            dataset_parameters: dict = None,
            dataset_checks: Union[
                List[BaseCheck], BaseSuite] = data_integrity(),
            train_test_datasets_checks: Union[
                List[BaseCheck], BaseSuite] = Suite(
                'Checks train test', train_test_validation())
    ) -> None:

        if dataset_parameters:
            if 'label' in dataset_parameters:
                value_label = dataset_parameters['label']
                logging.warning(f'label value {value_label} will be ignored')
                del dataset_parameters['label']
            self.dataset_parameters = dataset_parameters
        else:
            self.dataset_parameters = {}

        self.name = name

        self.whole_dataset_checks_suite = self.__generate_suite(
            dataset_checks,
            'Checks on whole dataset'
        )

        self.train_test_checks_suite = self.__generate_suite(
            train_test_datasets_checks,
            'Checks on train and test datasets'
        )

    def __call__(self, project: Project) -> None:
        """
        Run suite of checks on the project data.

        Parameters
        ----------
        project: project
        """

        self.__generate_datasets(project, **self.dataset_parameters)
        self.dataset_checks_results = self.whole_dataset_checks_suite.run(
            self.__dataset
        )
        self.train_test_checks_results = self.train_test_checks_suite.run(
            train_dataset=self.__train_dataset,
            test_dataset=self.__test_dataset
        )
        for results in [self.train_test_checks_results,
                        self.dataset_checks_results]:
            logger.logger.log_artifact(results, f'{results.name}')

    def __generate_datasets(self, project: Project, **kwargs) -> None:
        """
        Generate :class:`deepchecks.Dataset`


        Parameters
        ----------
        project: project
            :class:`~palma.Project`
        """

        df = pd.concat([project.X, project.y], axis=1)
        df.columns = [*project.X.columns.to_list(), "target"]
        self.__dataset = Dataset(df, label=project.y.name, **kwargs)

        self.__train_dataset = self.__dataset.copy(
            df.loc[project.validation_strategy.train_index])
        self.__test_dataset = self.__dataset.copy(
            df.loc[project.validation_strategy.test_index])

    @staticmethod
    def __generate_suite(
            checks: Union[List[BaseCheck], BaseSuite],
            name: str
    ) -> Suite:
        """
        Generate a Suite of checks from a list of checks or a suite of checks

        Parameters
        ----------
        checks: Union[List[BaseCheck], BaseSuite], optional
            List of checks or suite of checks
        name: str
            Name for the suite to returned

        Returns
        -------
        suite: :class:`deepchecks.Suite`
            instance of :class:`deepchecks.Suite`
        """

        if isinstance(checks, list) and all(
                isinstance(x, BaseCheck) for x in checks):
            suite = Suite(name, *checks)
        elif isinstance(checks, BaseSuite):
            suite = checks
            suite.name = name
        else:
            raise TypeError(
                "checks must be a list of instances of "
                "class BaseCheck or an instance of class Suite"
            )

        return suite

    def __str__(self) -> str:
        return "DeepCheck"


class Leakage(ProjectComponent):
    def __call__(self, project: Project) -> None:
        z = utils.get_splitting_matrix(
            project.X,
            project.validation_strategy.indexes_train_test)
        z = z == 2
