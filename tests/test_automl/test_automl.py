from sklearn.model_selection import ShuffleSplit

from palma.automl.automl import AutoMl


def test_automl_classification(classification_data):
    x, y = classification_data
    automl = AutoMl(
        "my-project",
        'classification',
        x, y,
        ShuffleSplit(),
    ).run("FlamlOptimizer", {"time_budget": 5})

    assert hasattr(automl, "project")
    assert hasattr(automl, "model")


def test_automl_regression(regression_data):
    x, y = regression_data
    automl = AutoMl(
        "my-project",
        'regression',
        x, y,
        ShuffleSplit(),
    ).run("FlamlOptimizer", {"time_budget": 5})

    assert hasattr(automl, "project")
    assert hasattr(automl, "model")
