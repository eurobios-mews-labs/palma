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

import tempfile

import pandas as pd
import pytest
from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from palma import ModelEvaluation
from palma import Project
from palma.components import FileSystemLogger
from palma.components import dashboard
from palma.components import performance


@pytest.fixture(scope='module')
def classification_data():
    X, y = make_classification(random_state=0, n_samples=300)
    return pd.DataFrame(X[:, :4]), pd.Series(y, name="target")


@pytest.fixture(scope='module')
def regression_data():
    X, y = make_regression()
    return pd.DataFrame(X[:, :4]), pd.Series(y)


@pytest.fixture(scope="module")
def classification_project(classification_data):
    from palma import set_logger
    set_logger(FileSystemLogger(tempfile.gettempdir()))
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(y)
    project = Project(problem="classification", project_name="test")

    project.start(
        X, y,
        splitter=model_selection.ShuffleSplit(n_splits=4, random_state=42))
    return project


@pytest.fixture(scope="module")
def unbuilt_classification_project():
    project = Project(problem="classification", project_name="test")
    return project


@pytest.fixture(scope='module')
def learning_data(classification_project, classification_data):
    X, y = classification_data
    estimator = RandomForestClassifier()

    learn = ModelEvaluation(estimator)
    learn.fit(classification_project)
    return classification_project, learn, X, y


@pytest.fixture(scope='module')
def get_scoring_analyser(learning_data):
    project, model, X, y = learning_data
    perf = performance.ScoringAnalysis(on="indexes_train_test")
    perf._add(project, model)

    perf.compute_metrics(metric={
        metrics.roc_auc_score.__name__: metrics.roc_auc_score,
        metrics.roc_curve.__name__: metrics.roc_curve
    })
    return perf


@pytest.fixture(scope='module')
def get_shap_analyser(learning_data):
    project, model, X, y = learning_data
    perf = performance.ShapAnalysis(on="indexes_val", n_shap=100,
                                    compute_interaction=True)
    perf(project, model)

    return perf


@pytest.fixture(scope='module')
def learning_data_regression(regression_data):
    from palma import set_logger
    set_logger(FileSystemLogger(tempfile.gettempdir()))

    estimator = Pipeline(
        steps=[("scaler", StandardScaler()), ("est", LinearRegression())])
    X, y = regression_data
    X = pd.DataFrame(X)
    y = pd.Series(y)
    project = Project(problem="regression", project_name="test")

    project.start(
        X, y,
        splitter=model_selection.ShuffleSplit(n_splits=4, random_state=42),
    )
    learn = ModelEvaluation(estimator)
    learn.fit(project)
    return project, learn, X, y


@pytest.fixture(scope='module')
def get_regression_analyser(learning_data_regression):
    project, model, X, y = learning_data_regression
    perf = performance.RegressionAnalysis(
        on="indexes_train_test")
    perf(project, model)
    perf.compute_metrics(metric={
        metrics.r2_score.__name__: metrics.r2_score,
    })

    return perf





@pytest.fixture(scope='module')
def build_classification_project(unbuilt_classification_project,
                                 classification_data):
    from palma import set_logger
    set_logger(FileSystemLogger(uri=tempfile.gettempdir()))
    project = Project(problem="classification", project_name="test")
    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(y)
    project.start(
        X,
        y,
        splitter=ShuffleSplit(n_splits=2)
    )
    return project


@pytest.fixture(scope='module')
def get_explainer_dashboard(classification_project):
    estimator = RandomForestClassifier()

    model = ModelEvaluation(estimator)
    model.add(dashboard.ExplainerDashboard(n_sample=100))
    model.fit(classification_project)
    return model.components["ExplainerDashboard"]
