import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import shap

from palma.components.logger import logger
from palma.components.performance import Analyser


class ShapAnalysis(Analyser):

    def __init__(self, on, n_shap, compute_interaction=False):
        super().__init__(on)
        self.n_shap = n_shap
        self.compute_interaction = compute_interaction

    def __call__(self, project: "Project", model: "ModelEvaluation"):
        self._add(project, model)
        self._compute_shap_values(n=self.n_shap,
                                  is_regression=self._is_regression,
                                  compute_interaction=self.compute_interaction)
        plot.figure()
        self.plot_shap_summary_plot()
        plot.figure()
        self.plot_shap_decision_plot()
        return self

    def _compute_shap_values(
            self, n, is_regression,
            compute_interaction=False
    ):

        i_loc: list = []  # list of all index
        self.shap_values = np.array([])
        self.shap_X = pd.DataFrame()
        self.shap_interaction = np.array([])
        self.shap_expected_value = 0
        explainer_method = _select_explainer(self.only_estimators[0])
        sizes = np.diff(np.linspace(0, n, num=len(self.indexes) + 1, dtype=int))
        for i, (train, test) in enumerate(self.indexes):

            i_loc_ = np.random.choice(test, size=sizes[i])
            if self.preproc_estimators:
                x_processed = pd.DataFrame(
                    self.preproc_estimators[i].transform(self.X.iloc[i_loc_]),
                    index=self.X.iloc[i_loc_].index, columns=self.X.columns)
            else:
                x_processed = self.X.iloc[i_loc_]
            if "linear" in str(explainer_method).lower():
                kwargs = dict(masker=x_processed)
            else:
                kwargs = dict()
            explainer = explainer_method(self.only_estimators[i], **kwargs)
            shap_values = explainer.shap_values(x_processed)
            shap_e_value = explainer.expected_value
            if compute_interaction and hasattr(explainer_method, "shap_interaction_values"):
                shap_interaction = explainer.shap_interaction_values(
                    x_processed)
                shap_interaction = shap_interaction[1]
                if self.shap_values.__len__() == 0:
                    self.shap_interaction = shap_interaction
                else:
                    self.shap_interaction = np.concatenate(
                        (self.shap_interaction, shap_interaction))

            if not is_regression and 'XGB' not in str(self.estimators[0]) and len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 1]
                shap_e_value = shap_e_value[1]

            if self.shap_values.__len__() == 0:
                self.shap_values = shap_values
                self.shap_expected_value = shap_e_value
            else:
                self.shap_values = np.concatenate(
                    (self.shap_values, shap_values))
                self.shap_expected_value += shap_e_value
            i_loc += list(i_loc_)
            self.shap_expected_value /= len(self.indexes)
            self.shap_X = pd.concat((self.shap_X, x_processed))
        self.__change_features_name_to_string()

    def __change_features_name_to_string(self):
        self.shap_X.columns = [str(c) for c in self.shap_X.columns]

    def plot_shap_summary_plot(self):
        shap.summary_plot(self.shap_values, self.shap_X)
        logger.logger.log_artifact(plot.gcf(), 'performance_shap_summary_plot')

    def plot_shap_decision_plot(self, **kwargs):
        shap.decision_plot(self.shap_expected_value, self.shap_values,
                           self.shap_X, **kwargs)

        logger.logger.log_artifact(plot.gcf(), 'performance_shap_decision_plot')

    def plot_shap_interaction(self, feature_x, feature_y):
        shap.dependence_plot(
            (feature_x, feature_y),
            self.shap_interaction, self.shap_X,
            display_features=self.shap_X
        )
        logger.logger.log_artifact(plot.gcf(), f"performance_shap_interaction_"
                                               f"{feature_x}_{feature_y}")

def _select_explainer(estimator):
    from sklearn import ensemble, linear_model
    from palma.base.model import get_estimator_name
    name = get_estimator_name(estimator)
    if name in [*ensemble.__dict__["__all__"], "XGBClassifier", "XGBRegressor"]:
        explainer_method = shap.TreeExplainer
    elif name in linear_model.__dict__["__all__"]:
        explainer_method = shap.LinearExplainer
    else:
        explainer_method = shap.TreeExplainer
    return explainer_method
