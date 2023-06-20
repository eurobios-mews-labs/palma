# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import pytest

from palma.base.engine import AutoSklearnOptimizer, BaseOptimizer, FlamlOptimizer


@pytest.fixture
def get_engine_autosklearn_classification(classification_data):
    X, y = classification_data
    engine = AutoSklearnOptimizer(
        problem="classification",
        engine_parameters=dict(time_left_for_this_task=10),
    )
    engine.optimize(X, y)
    return engine


@pytest.fixture
def get_engine_autosklearn_regression(regression_data):
    X, y = regression_data
    engine = AutoSklearnOptimizer(
        problem="classification",
        engine_parameters=dict(time_left_for_this_task=10),
    )
    engine.optimize(X, y)
    return engine


@pytest.fixture
def get_engine_flaml_regression(regression_data):
    X, y = regression_data
    engine = FlamlOptimizer(
        problem="regression",
        engine_parameters=dict(
            time_budget=5,
            estimator_list=["xgboost"],
        )
    )
    engine.optimize(X, y)
    return engine


def test_engine_flaml_set_problem():
    engine = FlamlOptimizer(
        problem="regression",
        engine_parameters=dict(
            time_budget=5,
            estimator_list=["xgboost"],
            task="unknown"))
    assert engine.engine_parameters["task"] == "regression", \
        "Problem was not correctly set"


# def test_engine_classification_optimize(get_engine_autosklearn_classification):
#     assert hasattr(get_engine_autosklearn_classification, 'optimize'), \
#         "Optimizer instance should have a optimize attribute"
#
#
# def test_engine_classification_optimizer(get_engine_autosklearn_classification):
#     assert hasattr(get_engine_autosklearn_classification, 'optimizer'),\
#         "Optimizer instance should have a optimizer attribute"
#
#
# def test_engine_classification_transformer(
#         get_engine_autosklearn_classification):
#     assert hasattr(get_engine_autosklearn_classification, 'transformer_'),\
#         "Optimizer instance should have a transformer_ attribute"
#
#
# def test_engine_classification_estimator(
#         get_engine_autosklearn_classification):
#     assert hasattr(get_engine_autosklearn_classification, 'estimator_'),\
#         "Optimizer instance should have a estimator_ attribute"
#
#
# def test_autosklearn_optimizer(get_engine_autosklearn_classification):
#     from autosklearn.classification import AutoSklearnClassifier
#     assert isinstance(
#         get_engine_autosklearn_classification.optimizer,
#         AutoSklearnClassifier
#     ), 'Optimizer should be an instance of AutoSklearnClassifier'
#
#
# def test_engine_classification_unknown_problem(regression_data):
#     engine = AutoSklearnOptimizer(
#         problem="unknown",
#         engine_parameters=dict(time_left_for_this_task=10),
#     )
#     with pytest.raises(ValueError) as exc_info:
#         engine.optimize(regression_data.X_train, regression_data.y_train)
#     assert type(exc_info.value) == ValueError, "Unknown problem name"


@pytest.fixture()
def get_dummy_engine():
    BaseOptimizer.__abstractmethods__ = set()

    class Dummy(BaseOptimizer):
        pass

    dummy = Dummy(engine_parameters=dict())
    return dummy


def test_dummy_engine_optimize(get_dummy_engine, learning_data_regression):
    project, learn, X, y = learning_data_regression
    assert get_dummy_engine.optimize(
        X,
        y) is None, "method of abstract engine should return None"


def test_dummy_engine_optimizer(get_dummy_engine):
    assert get_dummy_engine.optimizer is None, \
        "method of abstract engine should return None"


def test_dummy_engine_parameters(get_dummy_engine):
    assert get_dummy_engine.engine_parameters == dict(), \
        "method of abstract engine should return None"


def test_engine_regression_optimize_flaml(get_engine_flaml_regression):
    assert hasattr(get_engine_flaml_regression, 'optimize'), \
        "Optimizer instance should have a optimize attribute"


def test_engine_regression_optimizer_flaml(get_engine_flaml_regression):
    assert hasattr(get_engine_flaml_regression, 'optimizer'), \
        "Optimizer instance should have a optimize attribute"


def test_engine_regression_tranformer_flaml(get_engine_flaml_regression):
    assert hasattr(get_engine_flaml_regression, 'transformer_'), \
        "Optimizer instance should have a transformer_ attribute"


def test_engine_regression_estimator_flaml(get_engine_flaml_regression):
    assert hasattr(get_engine_flaml_regression, 'estimator_'), \
        "Optimizer instance should have a optimize attribute"
