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
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit, GroupKFold

from palma import Project
from palma.base.splitting_strategy import ValidationStrategy


@pytest.fixture
def baseline_splitting_strategy_called() -> dict:
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)

    splitting_strategy = ValidationStrategy(splitter=ShuffleSplit())
    X, y = splitting_strategy(X, y)

    return {
        'split': splitting_strategy,
        'X': X,
        'y': y
    }


def test_splitting_strategy_has_train_index_attribute(
        baseline_splitting_strategy_called):
    assert hasattr(
        baseline_splitting_strategy_called['split'], 'train_index'), \
        "SplittingStrategy instance should have a train_index attribute"


def test_splitting_strategy_has_test_index_attribute(
        baseline_splitting_strategy_called):
    assert hasattr(
        baseline_splitting_strategy_called['split'],
        'test_index'), \
        "SplittingStrategy instance should have a test_index attribute"


def test_splitting_strategy_raise_key_error():
    return None  # TODO add this feature

def test_splitting_strategy_has_sort_by_attribute():
    return None  # TODO add this feature
    X, y = make_classification()
    X = pd.DataFrame(X)
    X.rename(columns={0: "col1"}, inplace=True)
    splitting_strategy = ValidationStrategy(
        splitter=ShuffleSplit(),
        sort_by="col1")
    assert hasattr(splitting_strategy, 'sort_by'), \
        "SplittingStrategy instance should have a sort_by attribute"


def test_str():
    splitting_strategy = ValidationStrategy(splitter=ShuffleSplit(
        n_splits=10, random_state=1,
    ))
    expected_value = "train/test split with args: {'n_splits': 10," \
                     " 'test_size': None, 'train_size': None, " \
                     "'random_state': 1, '_default_test_size': " \
                     "0.1}"

    assert str(splitting_strategy) == expected_value, \
        "__str__ returns wrong value"


def test_columns_spitter():
    return None  # TODO add this feature
    from sklearn.datasets import make_classification
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X["train"] = (3 * np.random.uniform(size=X.shape[0])).astype(int)
    X["test"] = X["train"]
    splitting_strategy = ValidationStrategy(splitter=["train", "test"],
                                            fold_as_test=1)
    X_, y_ = splitting_strategy(X, y)
    assert len(splitting_strategy.indexes_val) == 1, 'wrong index building'
    assert len(splitting_strategy.indexes_val[0]) == 2, 'wrong index building'
    assert "train" not in X_.columns, 'fail to remove index column'
    indexes_train = X.index[X["train"] == 1]
    assert np.mean(
        splitting_strategy.train_index == indexes_train) == 1, "wrong test index"


def test_splitter_association():
    return None  # TODO add this feature
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)
    splitter = [([0, 1], [0, 2]), ([0, 1, 4], [0, 1, 45])]
    splitting_strategy = ValidationStrategy(
        splitter=splitter)
    X, y = splitting_strategy(X, y)
    for train, test in splitting_strategy.indexes_val:
        assert X.iloc[train].shape[0] == 2

    assert np.mean(splitting_strategy.train_index == splitter[-1][0]) == 1, \
        "wrong test index"


def test_group_splitting_strategy(classification_data):
    project = Project(project_name="test", problem="classification")
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(np.ravel(y))
    n_splits = 5
    groups = (np.random.uniform(size=y.__len__()) * 10).astype(int)
    project.start(
        X,
        y,
        splitter=GroupKFold(n_splits=n_splits),
        groups=groups
    )
    assert len(project.validation_strategy.indexes_val) == n_splits, \
        "wrong number of split"


def test_splitting_strategy(classification_data):
    project = Project(project_name="test", problem="classification")
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(np.ravel(y))
    n_splits = 5
    project.start(
        X,
        y,
        X_test=X,
        y_test=y,
        splitter=ShuffleSplit(n_splits=n_splits),
    )
    assert len(project.validation_strategy.indexes_val) == n_splits, \
        "wrong number of split"


def test_splitting_strategy_with_flaml_engine(classification_data):
    from palma import ModelSelector
    project = Project(project_name="test", problem="classification")
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(np.ravel(y))
    n_splits = 5
    groups = (np.random.uniform(size=y.__len__()) * 10).astype(int)
    project.start(
        X,
        y,
        splitter=GroupKFold(n_splits=n_splits),
        groups=groups
    )
    assert groups is not None
    assert project.validation_strategy.groups is not None
    ms = ModelSelector(engine="FlamlOptimizer",
                       engine_parameters={'time_budget': 5})
    ms.start(project)
    assert True
