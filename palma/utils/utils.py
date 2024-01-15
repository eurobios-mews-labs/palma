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

import hashlib
import json
import pickle
from copy import deepcopy
from datetime import datetime
from functools import wraps
from hashlib import sha256
from typing import Callable

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.pipeline import Pipeline


class AverageEstimator:
    """
    A simple ensemble estimator that computes the average prediction of a list of estimators.

    Parameters
    ----------
    estimator_list : list
        A list of individual estimators to be averaged.

    Attributes
    ----------
    estimator_list : list
        The list of individual estimators.
    n : int
        The number of estimators in the list.

    Methods
    -------
    predict(*args, **kwargs)
        Compute the average prediction across all estimators.

    predict_proba(*args, **kwargs)
        Compute the average class probabilities across all estimators.

    Returns
    -------
    numpy.ndarray
        The averaged prediction or class probabilities.
    """

    def __init__(self, estimator_list: list):
        """
        Initialize the AverageEstimator.

        Parameters
        ----------
        estimator_list : list
            A list of individual estimators to be averaged.
        """
        self.estimator_list = estimator_list
        self.n = len(estimator_list)

    def predict(self, *args, **kwargs) -> iter:
        est = 0
        for estimator in self.estimator_list:
            est += estimator.predict(*args, **kwargs)
        return np.array(est / self.n)

    def predict_proba(self, *args, **kwargs) -> iter:
        est = 0
        for estimator in self.estimator_list:
            est += estimator.predict_proba(*args, **kwargs)
        return np.array(est / self.n)


def _clone(estimator):
    """
    Create and return a clone of the input estimator.

    Parameters
    ----------
    estimator : object
        The estimator object to be cloned.

    Returns
    -------
    object
        A cloned copy of the input estimator.

    Notes
    -----
    This function attempts to create a clone of the input estimator using the
    `clone` function. If the `clone` function is not available or raises a
    `TypeError`, it falls back to using `deepcopy`. If both methods fail, the
    original estimator is returned.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> original_estimator = LinearRegression()
    >>> cloned_estimator = _clone(original_estimator)
    """
    try:
        return clone(estimator)
    except TypeError:
        pass
    try:
        return deepcopy(estimator)
    except TypeError:
        pass
    return estimator


def get_splitting_matrix(X: pd.DataFrame,
                         iter_cross_validation: iter,
                         expand=False) -> pd.DataFrame:
    """
    Generate a splitting matrix based on cross-validation iterations.

    Parameters
    ----------
    X : pd.DataFrame
        The input dataframe.
    iter_cross_validation : Iterable
        An iterable containing cross-validation splits (train, test).
    expand : bool, optional
        If True, the output matrix will have columns for both train and test
        splits for each iteration. If False (default), the output matrix will
        have columns for each iteration with 1 for train and 2 for test.

    Returns
    -------
    pd.DataFrame
        A matrix indicating the train (1) and test (2) splits for each
        iteration. Rows represent data points, and columns represent iterations.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
    ...                   'feature2': ['A', 'B', 'C', 'D', 'E']})
    >>> iter_cv = [(range(3), range(3, 5)), (range(2), range(2, 5))]
    >>> get_splitting_matrix(X, iter_cv)
    """
    if not expand:
        all_splits = pd.DataFrame(0, index=range(len(X)),
                                  columns=range(len(iter_cross_validation)))
        for i, (train, test) in enumerate(iter_cross_validation):
            all_splits.loc[train, i] = 1
            all_splits.loc[test, i] = 2
    else:
        all_splits = pd.DataFrame(False, index=range(len(X)),
                                  columns=range(2*len(iter_cross_validation)))
        for i, (train, test) in enumerate(iter_cross_validation):
            all_splits.loc[train, 2*i] = True
            all_splits.loc[test, 2*i + 1] = True

    return all_splits


def check_splitting_strategy(X: pd.DataFrame,
                             iter_cross_validation: iter):
    all_splits_train = pd.DataFrame(0, index=range(len(X)),
                                    columns=range(len(iter_cross_validation)))
    all_splits_test = pd.DataFrame(0, index=range(len(X)),
                                   columns=range(len(iter_cross_validation)))

    for i, (train, test) in enumerate(iter_cross_validation):
        all_splits_train.loc[train, i] = 1
        all_splits_test.loc[test, i] = 1

    m_train = all_splits_train.mean().mean()
    m_test = all_splits_test.mean().mean()
    print(f"Mean number of time an observation is used in "
          f"training set : {m_train}"
          f"\n"
          f"Mean number of time an observation is used in "
          f"testing set : {m_test}"
          )
    return all_splits_test, all_splits_train


def hash_dataframe(data: pd.DataFrame, how="whole"):
    if how == "whole":
        return hashlib.md5(
            pd.util.hash_pandas_object(
                data).values).hexdigest()
    elif how == "row_wise":
        return pd.DataFrame(
            np.array(
                [hashlib.md5(x).hexdigest() for x in
                 pd.util.hash_pandas_object(data).values])).T
    elif how == "types":
        return hashlib.md5(
            pd.util.hash_pandas_object(
                data.dtypes).values).hexdigest()
    else:
        raise TypeError(f"method {how} is unknown")


def get_hash(**kwargs) -> str:
    """ Return a hash of parameters """

    hash_ = sha256()
    for key, value in kwargs.items():
        if isinstance(value, datetime):
            hash_.update(str(kwargs[key]).encode('utf-8'))
        else:
            hash_.update(json.dumps(kwargs[key]).encode())

    return hash_.hexdigest()


def get_estimator_name(estimator) -> str:
    if hasattr(estimator, "steps"):
        est = estimator.steps[-1][1]
    else:
        est = estimator
    if hasattr(est, "__name__"):
        estimator_name = est.__name__
    else:
        estimator_name = str(est).split("(")[0]
    return estimator_name


def check_started(message: str, need_build: bool = False) -> Callable:
    """
    check_built is a decorator used for methods that must be called on \
    built or unbuilt :class:`~palma.Project`.
    If the :class:`~palma.Project` is_built attribute has \
    not the correct value, an AttributeError is raised with the message passed \
    as argument.

    Parameters
    ----------
    message: str
        Error message
    need_build: bool
        Expected value for :class:`~palma.Project` is_built \
        attribute

    Returns
    -------
    Callable
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
                project: 'Project',
                *args,
                **kwargs
        ) -> Callable:
            if project.is_started == need_build:
                return func(project, *args, **kwargs)
            else:
                raise AttributeError(message)

        return wrapper

    return decorator


def interpolate_roc(roc_curve_metric: dict[dict[tuple[dict[np.array]]]],
                    mean_fpr=np.linspace(0, 1, 100)):
    from numpy import interp
    roc_curve_interp = {}

    for i, _ in enumerate(roc_curve_metric.keys()):
        roc_curve_interp[i] = {}
        for step in ["train", "test"]:
            fpr, tpr, th = roc_curve_metric[i][step]

            tpr = interp(mean_fpr, fpr, tpr, left=True)
            th = interp(mean_fpr, fpr, th, left=True)
            roc_curve_interp[i][step] = mean_fpr, tpr, th
    return roc_curve_interp


def _get_processing_pipeline(estimators: list):
    if hasattr(estimators[0], "steps") and estimators[0].steps.__len__() > 1:
        processing_estimators = [Pipeline(est.steps[:-1]) for est in
                                 estimators]
        model_estimators = [est.steps[-1][1] for est in estimators]
    else:
        processing_estimators = []
        model_estimators = [est for est in estimators]
    return processing_estimators, model_estimators


def _get_and_check_var_importance(estimator):
    if hasattr(estimator, 'feature_importances_'):
        return estimator.feature_importances_
    if hasattr(estimator, 'coef_'):
        return estimator.coef_
    if hasattr(estimator, '__getitem__'):
        if hasattr(estimator[-1], 'feature_importances_'):
            return estimator[-1].feature_importances_
        if hasattr(estimator[-1], 'coef_'):
            return estimator[-1].coef_
