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

from palma.base.project import Project
from palma.components.base import Component
from palma.utils.utils import get_hash, AverageEstimator, _clone


class ModelEvaluation:
    def __init__(self, estimator):
        self.__date = datetime.now()
        self.__model_id = get_hash(date=self.__date)
        self.__estimator = estimator
        self.__components = {}
        self.estimator_name = get_estimator_name(estimator)
        self.metrics = {}

    def add(self, component):
        if isinstance(component, Component):
            self.__components.update({str(component): component})
        else:
            raise TypeError(
                "The added component must be an instance of class Component"
            )

    def fit(self, project: Project):
        ret = []
        for indexes in [project.validation_strategy.indexes_val,
                        project.validation_strategy.indexes_train_test
                        ]:
            ret.append(self.__get_fit_estimators(
                project.X, project.y,
                indexes))

        self.all_estimators_val_, self.avg_estimator_val_ = ret[0]
        self.all_estimators_, self.avg_estimator_ = ret[1]

        self.predictions_ = self.__compute_predictions(
            project, project.validation_strategy.indexes_train_test)
        self.predictions_val_ = self.__compute_predictions(
            project, project.validation_strategy.indexes_val)

        if hasattr(project, "_logger"):
            self._logger = project._logger

        for name, comp in self.__components.items():
            comp(project, self)

    def __get_fit_estimators(self, X, y, indexes):
        est = []
        for i, (train, _) in enumerate(indexes):
            x_train = X.iloc[train]
            y_train = y.iloc[train]
            self.__estimator = _clone(self.__estimator)
            self.__estimator.fit(x_train, y_train)
            est.append(self.__estimator)
        avg_estimator = AverageEstimator(est)
        return est, avg_estimator

    def __compute_predictions(self, project, indexes):
        predictions = {}
        for i, (train, test) in enumerate(indexes):
            est = self.all_estimators_val_[i]
            if hasattr(est, "predict_proba"):
                predict_train = est.predict_proba(project.X.iloc[train])[:, 1]
                predict_test = est.predict_proba(project.X.iloc[test])[:, 1]
            else:
                predict_train = est.predict(project.X.iloc[train])
                predict_test = est.predict(project.X.iloc[test])

            predictions[i] = dict(train=predict_train, test=predict_test)
        return predictions

    @property
    def id(self) -> str:
        return self.__model_id

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            raise AttributeError(
                "Logger not found, try to add a logger before starting your "
                "project")

        return self._logger

    @property
    def components(self):
        return self.__components


def get_estimator_name(estimator):
    if hasattr(estimator, "steps"):
        est = estimator.steps[-1][1]
    else:
        est = estimator
    if hasattr(est, "__name__"):
        estimator_name = est.__name__
    else:
        estimator_name = str(est).split("(")[0]
    return estimator_name
