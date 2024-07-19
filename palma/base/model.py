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
import numpy as np
import pandas as pd

from palma.base.project import Project
from palma.components.base import Component
from palma.utils.utils import (get_hash, AverageEstimator, _clone,
                               get_estimator_name)


class ModelEvaluation:
    def __init__(self, estimator):
        self.__date = datetime.now()
        self.__model_id = get_hash(date=self.__date)
        self.__estimator = estimator
        self.__components = {}
        self.estimator_name = get_estimator_name(estimator)
        self.metrics = {}

    def add(self, component, name=None):
        if name is None:
            name = str(component)
        # FIXME : raise error when component exists already
        if isinstance(component, Component):
            self.__components.update({name: component})
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
            project, project.validation_strategy.indexes_train_test, self.all_estimators_)
        self.predictions_val_ = self.__compute_predictions(
            project, project.validation_strategy.indexes_val, self.all_estimators_val_, val=True)

        self.__project = project
        self.__all_estimators = self.all_estimators_

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

    def __compute_predictions(self, project, indexes, estimators, val=False):
        predictions = {}
        n_y = project.y.shape[0]
        n_class = project.y.unique().shape[0]
        record_split_type = pd.Series(np.zeros(n_y))
        record_split_type = record_split_type.astype(str)
        record_split_type.loc[:] = ""
        # ------------------- predict -----------------------
        if val:
            self.__val_predict_df = pd.Series(np.zeros(n_y))
        else:
            self.__predict_df = pd.Series(np.zeros(n_y))
            self.__predict_df = pd.Series(np.zeros(n_y))
        # ---------------- predict_proba --------------------
        est = estimators[0]
        if hasattr(est, "predict_proba"):
            if val:
                self.__val_predict_proba_df = pd.DataFrame(np.zeros((n_y, n_class)))
            else:
                self.__predict_proba_df = pd.DataFrame(np.zeros((n_y, n_class)))
                self.__predict_proba_df = self.__predict_proba_df.copy()
        # ============== COMPUTE PREDICTIONS ===============
        for i, (train, test) in enumerate(indexes):
            if val:
                record_split_type.loc[train] += "."
            else:
                record_split_type.loc[train] += "train"
            record_split_type.loc[test] += "test"
            est = estimators[i]
            if hasattr(est, "predict_proba"):
                pred_train = est.predict_proba(project.X.iloc[train])
                pred_test = est.predict_proba(project.X.iloc[test])
                predict_train = pred_train[:, 1]
                predict_test = pred_test[:, 1]
                if val:
                    # ---------------- predict_proba --------------------
                    self.__val_predict_proba_df.loc[train, :] = pred_train
                    self.__val_predict_proba_df.loc[test, :] = pred_test
                    # -------------------- predict ----------------------
                    self.__val_predict_df.loc[train] = est.predict(project.X.iloc[train])
                    self.__val_predict_df.loc[test] = est.predict(project.X.iloc[test])
                else:
                    # ---------------- predict_proba --------------------
                    self.__predict_proba_df.loc[train, :] = pred_train
                    self.__predict_proba_df.loc[test, :] = pred_test
                    # -------------------- predict ----------------------
                    self.__predict_df.loc[train] = est.predict(project.X.iloc[train])
                    self.__predict_df.loc[test] = est.predict(project.X.iloc[test])
            else:
                # -------------------- predict ----------------------
                predict_train = est.predict(project.X.iloc[train])
                predict_test = est.predict(project.X.iloc[test])
                if val:
                    self.__val_predict_df.loc[train] = predict_train
                    self.__val_predict_df.loc[test] = predict_test
                else:
                    self.__predict_df.loc[train] = predict_train
                    self.__predict_df.loc[test] = predict_test
            predictions[i] = dict(train=predict_train, test=predict_test)
        # =============== final formatting =============
        if val:
            self.__val_predict_df = self.__val_predict_df.to_frame()
            self.__val_predict_df["split_type"] = record_split_type
            self.__val_predict_df.index = project.y.index
        else:
            self.__predict_df = self.__predict_df.to_frame()
            self.__predict_df["split_type"] = record_split_type
            self.__predict_df.index = project.y.index
        if hasattr(est, "predict_proba"):
            if val:
                self.__val_predict_proba_df["split_type"] = record_split_type
                self.__val_predict_proba_df.index = project.y.index
            else:
                self.__predict_proba_df["split_type"] = record_split_type
                self.__predict_proba_df.index = project.y.index
        else:
            self.__predict_proba_df = None
            self.__val_predict_proba_df = None
        return predictions

    def predict(self, X, return_df=False):
        """
        Mimic sklearn predict method. Note that contrary to fit,
        This function does not perform cross validation prediction
        nor does it train the estimators.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        return_df : True = returns pd.Series, False = return np.array.

        Returns
        -------
        C : array, shape (n_samples,) or pandas series, shape (n_samples)
            or None if unmatched features with train dataframe.
            Returns predicted values.
        """
        returned_obj = None
        if f"{list(X.columns)}" != f"{list(self.__project.X.columns)}":
            print("Error: The dataframe doesn't have the same number of predictors or the same"
                  " column labels as the train dataframe.")
        else:
            print("Warning: Fit method has already created train, validation and test predictions"
                  " based on the project data. Make sure that running predict is relevant in your case,"
                  " such as for testing new data. Predictions from fitted object can be accessed"
                  " by calling the following attributes: .predictions_, .predict_df, .val_predict_df")
            n_y = self.__project.y.shape[0]
            self.__predict_df = pd.Series(np.zeros(n_y))
            self.__predict_proba_df = None
            n_est = len(self.__all_estimators)
            for est in self.__all_estimators:
                # -------------------- predict ----------------------
                predict_test = est.predict(X)
                self.__predict_df += predict_test/n_est
            self.__predict_df.index = self.__project.y.index
            if not return_df: returned_obj = self.__predict_df.values
            else: returned_obj = self.__predict_df
        return returned_obj

    def predict_proba(self, X, return_df=False):
        """
        Mimic sklearn predict_proba method. Note that contrary to fit,
        This function does not perform cross validation prediction
        nor does it train the estimators.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        return_df : True = returns pd.Dataframe, False = return np.array.

        Returns
        -------
        C : array, shape (n_samples, n_features)
            or pandas dataframe, shape (n_samples, n_features)
            or None if unmatched features with train dataframe.
            Returns predicted probabilities.
        """
        returned_obj = None
        if f"{list(X.columns)}" != f"{list(self.__project.X.columns)}":
            print("Error: The dataframe doesn't have the same number of predictors or the same"
                  " column labels as the train dataframe.")
        else:
            print("Warning: Fit method has already created train, validation and test predictions"
                  " based on the project data. Make sure that running predict_proba is relevant in your case,"
                  " such as for testing new data. Predictions from fitted object can be accessed"
                  " by calling the following attributes: .predictions_, .predict_proba_df, .val_predict_proba_df")
            n_y = self.__project.y.shape[0]
            n_class = self.__project.y.unique().shape[0]
            self.__predict_proba_df = pd.DataFrame(np.zeros((n_y,n_class)))
            self.__predict_proba_df.index = self.__project.y.index
            self.__predict_df = None
            # ============== COMPUTE PREDICTIONS ===============
            n_est = len(self.__all_estimators)
            for est in self.__all_estimators:
                predict_test = est.predict_proba(X)
                self.__predict_proba_df += predict_test / n_est
            if not return_df: returned_obj = self.__predict_proba_df.values
            else: returned_obj = self.__predict_proba_df
        return returned_obj

    @property
    def id(self) -> str:
        return self.__model_id

    @property
    def components(self):
        return self.__components

    @property
    def unfit_estimator(self):
        return self.__estimator

    @property
    def predict_df(self) -> pd.Series:
        return self.__predict_df

    @property
    def val_predict_df(self) -> pd.Series:
        return self.__val_predict_df

    @property
    def predict_proba_df(self) -> pd.DataFrame:
        if hasattr(self, "_ModelEvaluation__predict_proba_df"): df = self.__predict_proba_df
        else: df = None
        return df

    @property
    def val_predict_proba_df(self) -> pd.DataFrame:
        if hasattr(self, "_ModelEvaluation__val_predict_proba_df"):df = self.__val_predict_proba_df
        else: df = None
        return df
