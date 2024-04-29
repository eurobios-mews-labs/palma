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
import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics, model_selection
from sklearn.ensemble import RandomForestClassifier

from palma import ModelEvaluation, Project
from palma import set_logger
from palma.components import performance, FileSystemLogger

matplotlib.use("agg")


@pytest.fixture(scope='module')
def get_scoring_analyser(classification_data):
    set_logger(FileSystemLogger(tempfile.gettempdir() + "/logger"))

    X, y = classification_data
    X = pd.DataFrame(X)
    y = pd.Series(y)
    project = Project(problem="classification",
                      project_name=str(np.random.uniform()))

    project.start(
        X, y,
        splitter=model_selection.ShuffleSplit(n_splits=4, random_state=42))
    estimator = RandomForestClassifier()

    learn = ModelEvaluation(estimator)
    learn.fit(project)

    perf = performance.ScoringAnalysis(on="indexes_val")
    perf(project, learn)

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


def test_classification_perf(get_scoring_analyser):
    performance.plot.figure(figsize=(6, 6), dpi=200)
    get_scoring_analyser.plot_roc_curve(
        plot_method="beam",
        label="train")
    get_scoring_analyser.plot_roc_curve(
        plot_method="mean", mode="minmax")
    get_scoring_analyser.plot_roc_curve(
        plot_method="all")
    with pytest.raises(ValueError) as e:
        get_scoring_analyser.plot_roc_curve(
            plot_method="test", mode="minmax")
    assert str(e.value) == "argument plot_method=test is not recognize"
    with pytest.raises(ValueError) as e:
        get_scoring_analyser.plot_roc_curve(
            plot_method="mean", mode="test")
    assert str(e.value) == "Argument mode test is unknown"
    performance.plot.figure(figsize=(6, 6), dpi=200)
    get_scoring_analyser.variable_importance()
    get_scoring_analyser.plot_variable_importance(mode="boxplot")
    get_scoring_analyser.plot_variable_importance(mode="std")
    get_scoring_analyser.plot_variable_importance(mode="minmax")


def test_raise_value_when_no_threshold(get_scoring_analyser):
    with pytest.raises(AttributeError) as exc_info:
        get_scoring_analyser.confusion_matrix(in_percentage=True)
    assert exc_info.type == AttributeError, "Wrong error type"


def test_compute_threshold(get_scoring_analyser: performance.ScoringAnalysis):
    get_scoring_analyser._Analyser__metrics = {}
    get_scoring_analyser.compute_threshold("fpr", value=0.2)
    get_scoring_analyser.confusion_matrix(in_percentage=True)

    get_scoring_analyser.compute_threshold("optimize_metric",
                                           metric=metrics.f1_score)

    get_scoring_analyser.compute_threshold("total_population",
                                           metric=metrics.f1_score)
    get_scoring_analyser.plot_threshold()
    with pytest.raises(ValueError) as e:
        get_scoring_analyser.compute_threshold("test",
                                               metric=metrics.f1_score)
    assert str(e.value) == "method test is not recognized"

    with pytest.raises(ValueError) as e:
        get_scoring_analyser.compute_threshold("optimize_metric")
    assert str(e.value) == "Argument metric must not be not None"


def test_shap_scoring(get_shap_analyser):
    get_shap_analyser.plot_shap_interaction(
        get_shap_analyser.shap_X.columns[0],
        get_shap_analyser.shap_X.columns[1])


def test_shap_regression(get_shap_analyser):
    get_shap_analyser.plot_shap_summary_plot()
    get_shap_analyser.plot_shap_decision_plot()


def test_analyser_raise_error_parameters(
        get_shap_analyser, learning_data):
    project, model, X, y = learning_data
    get_shap_analyser.__init__(on="test", n_shap=50)
    with pytest.raises(ValueError) as e:
        get_shap_analyser._add(project, model)
    assert (str(e.value) == "on parameter : test is not understood."
                            " The possible values are 'indexes_train_test'"
                            " or 'indexes_val'")


def test_shap_regression_compute(get_shap_analyser):
    get_shap_analyser._compute_shap_values(100, is_regression=True,
                                           explainer_method="auto")


def test_regression_perf(get_regression_analyser):
    get_regression_analyser.plot_prediction_versus_real()
    get_regression_analyser.plot_variable_importance()
    get_regression_analyser.plot_errors_pairgrid()


def test_performance_pipeline_version(get_regression_analyser):
    pass  # TODO


def test_performance_get_metric_dataframe(get_regression_analyser):
    get_regression_analyser.compute_metrics(metric={
        metrics.r2_score.__name__: metrics.r2_score,
        metrics.mean_absolute_error.__name__: metrics.mean_absolute_error,
        metrics.mean_squared_error.__name__: metrics.mean_squared_error,
    })
    assert len(get_regression_analyser.get_test_metrics().columns) >= len(
        get_regression_analyser.metrics.keys())
    print(get_regression_analyser.get_train_metrics())
    assert get_regression_analyser.get_train_metrics()["r2_score"].iloc[0] < 0.5
    get_regression_analyser.plot_errors_pairgrid()


def test_shap_with_pipeline(get_shap_analyser, learning_data_regression):
    project, model, X, y = learning_data_regression
    get_shap_analyser.__init__(on="indexes_train_test", n_shap=50)
    get_shap_analyser._add(project, model)


def test_compute_metrics(get_regression_analyser):
    assert len(get_regression_analyser.metrics) > 5

    for v in ['max_error', 'mean_absolute_error', 'mean_squared_error',
              'mean_squared_log_error', 'median_absolute_error',
              'mean_absolute_percentage_error', 'mean_pinball_loss', 'r2_score',
              'explained_variance_score', 'mean_tweedie_deviance',
              'mean_poisson_deviance', 'mean_gamma_deviance',
              'd2_tweedie_score', 'd2_pinball_score',
              'd2_absolute_error_score']:
        assert v in get_regression_analyser.metrics.keys()


def test_permutation_feature_importance(learning_data):
    res_dir = tempfile.gettempdir() + "/logger"
    set_logger(FileSystemLogger(res_dir))
    project, model, X, y = learning_data
    perf = performance.PermutationFeatureImportance(scoring='roc_auc')
    perf(project, model)
    print(f"{os.listdir(res_dir) = }")
