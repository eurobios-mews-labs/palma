from sklearn.model_selection import ShuffleSplit

from palma.automl.automl import AutoMl


def test_automl_classification(classification_data):
    automl = AutoMl(
        "my-project",
        'classification',
        *classification_data,
        ShuffleSplit(),
    ).run("FlamlOptimizer", {"time_budget": 5})

    assert hasattr(automl, "project")
    assert hasattr(automl, "model")
