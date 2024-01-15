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
import os
from typing import Dict, Union

import numpy as np

from palma.components.base import Component
from palma.utils.utils import _get_processing_pipeline

try:
    from explainerdashboard import ExplainerDashboard as ExpDash
    from explainerdashboard.explainers import BaseExplainer
except (ImportError, ModuleNotFoundError):
    pass

default_config_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "configuration_materials",
    "default_dashboard.yaml"
)


class ExplainerDashboard(Component):

    def __init__(self,
                 dashboard_config: Union[
                     str, Dict] = default_config_path,
                 n_sample: int = None,
                 ):

        import yaml
        if isinstance(dashboard_config, Dict):
            self.dashboard_config = dashboard_config
        else:
            logging.info(f'loading config file : {dashboard_config}')
            with open(dashboard_config, 'r') as f:
                self.dashboard_config = yaml.safe_load(f)
        self.n_sample = n_sample

    def __call__(self, project: "Project", model: "Model") -> "ExpDash":
        """
        This function returns dashboard instance. This dashboard is to be run
        using its `run` method.

        Examples
        --------
        >>> db = ExpDash(dashboard_config="path_to_my_config")
        >>> explainer_dashboard = db(project, model)
        >>> explainer_dashboard.run(
        >>>    port="8050", host="0.0.0.0", use_waitress=False)

        Parameters
        ----------
        project: Project
            Instance of project used to compute explainer.
        model: Run
            Current run to use in explainer.
        """

        self.update_config(
            {"dashboard_parameters": {"title": project.project_name}}
        )
        explainer = self._get_explainer(project, model)
        dashboard = self._get_dashboard(explainer)
        return dashboard

    def update_config(self, dict_value: Dict[str, Dict]):
        """
        Update specific parameters from the actual configuration.

        Parameters
        ----------
        dict_value: dict
            explainer_parameters: dict
                Parameters to be used in see `explainerdashboard.RegressionExplainer`
                or `explainerdashboard.ClassifierExplainer`.
            dashboard_parameters: dict
                Parameters use to compose dashboard tab, items or themes
                for `explainerdashboard.ExplainerDashboard`.
                Tabs and component of the dashboard can be hidden, see
                `customize dashboard section <https://explainerdashboard.readthedocs.io/en/latest/custom.html>`_
                for more detail.

        Example
        -------
        >>> db.update_config({"explainer_parameters":{"shap": "tree"}})
        """
        for dashboard_component in dict_value.keys():
            for parameter in dict_value[dashboard_component].keys():
                self.dashboard_config[dashboard_component][parameter] = \
                    dict_value[dashboard_component][parameter]

    def _prepare_dataset(self) -> None:
        """
        This function performs the following processing steps :
            - Ensure that column name is str (bug encountered in dashboard)
            - Get code from categories just in case of category data types
            - Sample the data if specified by user
        """
        if self.n_sample is not None:
            idx = np.random.choice(range(len(self.__X_)), size=self.n_sample)
            self.__X_ = self.__X_.iloc[idx]
            self.__y_ = self.__y_.iloc[idx]

        col_cat = list(self.__X_.columns[self.__X_.dtypes == "category"])
        for col in col_cat:
            self.__X_[col] = self.__X_[col].cat.codes

        self.__X_.columns = [str(c) for c in self.__X_.columns]

    def _get_explainer(self,
                       project: "Project",
                       model: "Model"
                       ) -> "BaseExplainer":
        from explainerdashboard import ClassifierExplainer, RegressionExplainer
        processing_estimators, model_estimators = _get_processing_pipeline(
            model.all_estimators_)
        if len(model_estimators) > 1:
            raise ValueError("Explainer cannot be set with using "
                             "cross-validation")
        if processing_estimators:
            self.__X_ = processing_estimators[0].transform(
                project.X.iloc[project.validation_strategy.test_index]
            )

        else:
            self.__X_ = project.X.iloc[
                project.validation_strategy.test_index]
        self.__y_ = project.y.iloc[
            project.validation_strategy.test_index]
        self._prepare_dataset()
        explainer_kwargs = dict(
            X=self.__X_,
            y=self.__y_,
            model=model_estimators[0],
            **self.dashboard_config["explainer_parameters"])

        if project.problem == "classification":
            return ClassifierExplainer(**explainer_kwargs)
        if project.problem == "regression":
            return RegressionExplainer(**explainer_kwargs)

    def _get_dashboard(self,
                       explainer: "BaseExplainer") -> "ExplainerDashboard":

        dashboard = ExpDash(
            explainer,
            **self.dashboard_config["dashboard_parameters"]
        )
        return dashboard
