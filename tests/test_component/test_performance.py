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
import matplotlib
from sklearn import metrics

from palma.components import performance
import pytest
matplotlib.use("agg")


def test_classification_perf(get_scoring_analyser):
    performance.plot.figure(figsize=(6, 6), dpi=200)
    get_scoring_analyser.plot_roc_curve(
        plot_method="beam",
        label="train")
    get_scoring_analyser.plot_roc_curve(
        plot_method="mean", mode="minmax")
    with pytest.raises(ValueError) as e:
        get_scoring_analyser.plot_roc_curve(
            plot_method="test", mode="minmax")
    assert str(e.value) == "argument plot_method=test is not recognize"

    performance.plot.figure(figsize=(6, 6), dpi=200)
    get_scoring_analyser.variable_importance()
    get_scoring_analyser.plot_variable_importance(mode="boxplot")
    get_scoring_analyser.plot_variable_importance(mode="std")
    get_scoring_analyser.plot_variable_importance(mode="minmax")


def test_raise_value_when_no_threshold(get_scoring_analyser):
    with pytest.raises(AttributeError) as exc_info:
        get_scoring_analyser.confusion_matrix(in_percentage=True)
    assert exc_info.type == AttributeError, "Wrong error type"


def test_compute_threshold(get_scoring_analyser):
    get_scoring_analyser.metrics = {}
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
    get_shap_analyser._on = "test"
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
    assert get_regression_analyser.get_train_metrics()["r2_score"].iloc[0] < 0.2
    get_regression_analyser.plot_errors_pairgrid()

