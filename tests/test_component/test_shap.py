
import matplotlib
import pytest

from palma.components import ShapAnalysis

matplotlib.use("agg")

__n_shap__ = 13


@pytest.fixture(scope='module')
def get_shap_analyser(learning_data):
    project, model, X, y = learning_data
    perf = ShapAnalysis(on="indexes_val", n_shap=__n_shap__,
                        compute_interaction=True)
    perf(project, model)
    return perf


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
    get_shap_analyser._compute_shap_values(__n_shap__, is_regression=True)


def test_shap_scoring(get_shap_analyser):
    get_shap_analyser.plot_shap_interaction(
        get_shap_analyser.shap_X.columns[0],
        get_shap_analyser.shap_X.columns[1])


def test_shap_scoring(learning_data):
    project, model, X, y = learning_data
    perf = ShapAnalysis(
        on="indexes_train_test",
        n_shap=__n_shap__,
        compute_interaction=True)
    perf(project, model)
    perf.plot_shap_summary_plot()
    perf.plot_shap_decision_plot()

    assert perf.shap_values.shape == (__n_shap__, X.shape[1])
    assert perf.shap_X.shape == (__n_shap__, X.shape[1])
    assert isinstance(perf.shap_expected_value, float)


def test_shap_regression(learning_data_regression):
    project, model, X, y = learning_data_regression

    perf = ShapAnalysis(
        on="indexes_train_test",
        n_shap=__n_shap__,
        compute_interaction=True)
    perf(project, model)
    perf.plot_shap_summary_plot()
    perf.plot_shap_decision_plot()

    assert perf.shap_values.shape == (__n_shap__, X.shape[1])
    assert perf.shap_X.shape == (__n_shap__, X.shape[1])
    assert isinstance(perf.shap_expected_value, float)


def test_shap_with_pipeline(get_shap_analyser, learning_data_regression):
    project, model, X, y = learning_data_regression
    get_shap_analyser.__init__(on="indexes_train_test", n_shap=__n_shap__)
    get_shap_analyser._add(project, model)
