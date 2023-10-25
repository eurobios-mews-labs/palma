import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics
from sklearn.datasets import make_classification
from palma import Project, ModelEvaluation, ModelSelector
from palma.components import ScoringAnalysis, ShapAnalysis
from palma.utils import plotting


X, y = make_classification(n_informative=8, n_features=30, n_samples=1000)
X, y = pd.DataFrame(X), pd.Series(y).astype(bool)

project = Project(problem="classification", project_name="test")
project.start(
    X, y,
    splitter=model_selection.ShuffleSplit(n_splits=4, random_state=42),
)

plotting.plot_splitting_strategy(
    project.X, project.y,
    iter_cross_validation=project.validation_strategy.indexes_train_test, cmap="rainbow_r")
plt.tight_layout()

ms = ModelSelector(engine="FlamlOptimizer",
                   engine_parameters=dict(
                       time_budget=10,
                       estimator_list=['lgbm', 'rf', 'xgboost']))
ms.start(project)

model = ModelEvaluation(estimator=ms.best_model_)
model.add(ScoringAnalysis(on="indexes_train_test"))
model.add(ShapAnalysis(on="indexes_train_test", n_shap=100))
model.fit(project)

analyser = model.components["ScoringAnalysis"]
shap_analyser = model.components["ShapAnalysis"]

f, ax = plt.subplots(figsize=(5, 5))

analyser.plot_roc_curve()
analyser.compute_threshold(method="fpr", value=0.5)
analyser.confusion_matrix(in_percentage=True)

plt.figure(figsize=(6, 6))

analyser.plot_roc_curve()
analyser.plot_threshold(label="Threshold (FPR = 0.5)")
plt.legend(loc=4)


