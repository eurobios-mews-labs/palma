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

import typing
from abc import ABCMeta

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.metrics import _regression, _ranking
from sklearn.inspection import permutation_importance

from palma.components.base import ModelComponent
from palma.components.logger import logger
from palma.utils import plotting, utils

__fpr_sampling__ = np.linspace(0, 1, 200)


class Analyser(ModelComponent, metaclass=ABCMeta):
    """
    Analyser class for performing analysis on a model.

    Parameters
    ----------
    on : str
        The type of analysis to perform. Possible values are
        "indexes_train_test" or "indexes_val".
    """

    def __init__(self, on):
        self.__on = on
        self.__metrics = {}
        self._hidden_metrics = {}

    def __call__(self, project: "Project", model: "ModelEvaluation"):
        self._add(project, model)

    def _add(self, project, model):

        if self.__on == "indexes_train_test":
            self.indexes = project.validation_strategy.indexes_train_test
            self.estimators = model.all_estimators_
            self.predictions = model.predictions_

        elif self.__on == "indexes_val":
            self.indexes = project.validation_strategy.indexes_val
            self.estimators = model.all_estimators_val_
            self.predictions = model.predictions_val_
        else:
            raise ValueError(
                f"on parameter : {self.__on} is not understood. "
                f"The possible values are "
                f"'indexes_train_test' or 'indexes_val'")
        self.X = project.X
        self.y = project.y

        __tmp = utils._get_processing_pipeline(self.estimators)
        self.preproc_estimators, self.only_estimators = __tmp
        self._is_regression = project.problem == "regression"
        if project.problem == "regression":
            for metric in _regression.__ALL__:
                try:
                    self.compute_metrics({metric: getattr(_regression, metric)})
                except ValueError:
                    pass
        elif project.problem == "classification":
            for metric in ["roc_auc_score",
                           "label_ranking_average_precision_score",
                           "coverage_error", "label_ranking_loss", "dcg_score"]:
                try:
                    self.compute_metrics({metric: getattr(_ranking, metric)})
                except ValueError:
                    pass

    def variable_importance(self):
        """
        Compute the feature importance for each estimator.

        Returns
        -------
        feature_importance : pandas.DataFrame
            DataFrame containing the feature importance values for each estimator.
        """
        feature_importance = pd.DataFrame(columns=self.X.columns)
        for i, _ in enumerate(self.indexes):
            importance = utils._get_and_check_var_importance(
                self.estimators[i])
            feature_importance.loc[i, :] = importance
        return feature_importance.T

    def compute_metrics(self, metric: dict):
        """
        Compute the specified metrics for each estimator.

        Parameters
        ----------
        metric : dict
            Dictionary containing the metric name as key and the metric function as value.
        """
        from palma import logger
        for name, fun in metric.items():
            self._compute_metric(name, fun)

        for m_name, metric_fold in self.get_test_metrics().to_dict().items():
            for k, v in metric_fold.items():
                if isinstance(v, float) or isinstance(v, int):
                    logger.logger.log_metrics(
                        {f"{m_name}_fold{k}": v}, path="metrics")

    def _compute_metric(self, name: str, fun: typing.Callable):
        """
        Compute a specific metric and add it to the metrics attribute.

        Parameters
        ----------
        name : str
            The name of the metric.
        fun : callable
            The function to compute the metric.
        """
        self.__metrics[name] = {}
        for i, (train, test) in enumerate(self.indexes):
            self.__metrics[name][i] = dict(
                train=fun(self.y.iloc[train],
                          self.predictions[i]["train"]),
                test=fun(self.y.iloc[test],
                         self.predictions[i]["test"])
            )

    def get_train_metrics(self) -> pd.DataFrame:
        """
        Get the computed metrics for the training set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the computed metrics for the training set.
        """
        return self.__get_metrics_helper("train")

    def get_test_metrics(self) -> pd.DataFrame:
        """
        Get the computed metrics for the test set.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the computed metrics for the test set.
        """
        return self.__get_metrics_helper("test")

    def __get_metrics_helper(self, identifier) -> pd.DataFrame:
        m = self.__metrics
        ret = pd.DataFrame(columns=list(m.keys()),
                           index=m[list(m.keys())[0]].keys())
        for k in m.keys():
            for f in m[k].keys():
                ret.loc[f, k] = m[k][f][identifier]
        return ret

    def plot_variable_importance(self, mode="minmax",
                                 color="darkblue",
                                 cmap="flare", **kwargs):
        """
        Plot the variable importance.

        Parameters
        ----------
        mode : str, optional
            The mode for plotting the variable importance, by default "minmax".
        color : str, optional
            The color for the plot, by default "darkblue".
        cmap : str, optional
            The colormap for the plot, by default "flare".
        """
        from palma.utils.plotting import plot_variable_importance
        plot_variable_importance(
            self.variable_importance(), mode=mode, color=color, cmap=cmap,
            **kwargs
        )
        logger.logger.log_artifact(plot.gcf(), "variable_importance")

    @property
    def metrics(self):
        return self.__metrics


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
        return self

    def __select_explainer(self):
        from sklearn import ensemble, linear_model
        from palma.base.model import get_estimator_name
        name = get_estimator_name(self.estimators[0])
        if name in [*ensemble.__dict__["__all__"],
                    "XGBClassifier", "XGBRegressor"]:
            explainer_method = shap.TreeExplainer
        elif name in linear_model.__dict__["__all__"]:
            explainer_method = shap.LinearExplainer
        else:
            explainer_method = shap.TreeExplainer
        return explainer_method

    def _compute_shap_values(
            self, n, is_regression,
            explainer_method=shap.TreeExplainer,
            compute_interaction=False
    ):
        if explainer_method == "auto":
            explainer_method = self.__select_explainer()
        i_loc: list = []  # list of all index
        self.shap_values = np.array([])
        self.shap_X = pd.DataFrame()
        self.shap_interaction = np.array([])
        self.shap_expected_value = 0
        sizes = np.diff(np.linspace(0, n, num=len(self.indexes) + 1, dtype=int))
        for i, (train, test) in enumerate(self.indexes):

            i_loc_ = np.random.choice(test, size=sizes[i])
            if self.preproc_estimators:
                x_processed = pd.DataFrame(
                    self.preproc_estimators[i].transform(self.X.iloc[i_loc_]),
                    index=self.X.iloc[i_loc_].index, columns=self.X.columns)
            else:
                x_processed = self.X.iloc[i_loc_]
            explainer = explainer_method(self.only_estimators[i],
                                         masker=x_processed)
            shap_values = explainer.shap_values(x_processed)
            shap_e_value = explainer.expected_value
            if compute_interaction:
                shap_interaction = explainer.shap_interaction_values(
                    x_processed)
                if compute_interaction:
                    shap_interaction = shap_interaction[1]
                    if self.shap_values.__len__() == 0:
                        self.shap_interaction = shap_interaction
                    else:
                        self.shap_interaction = np.concatenate(
                            (self.shap_interaction, shap_interaction))

            if not is_regression and 'XGB' not in str(self.estimators[0]):
                shap_values = shap_values[1]
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
        import shap
        shap.summary_plot(self.shap_values, self.shap_X)

    def plot_shap_decision_plot(self, **kwargs):
        shap.decision_plot(self.shap_expected_value, self.shap_values,
                           self.shap_X, **kwargs)

    def plot_shap_interaction(self, feature_x, feature_y):
        shap.dependence_plot(
            (feature_x, feature_y),
            self.shap_interaction, self.shap_X,
            display_features=self.shap_X
        )
        logger.logger.log_artifact(plot.gcf(), f"shap_interaction_"
                                               f"{feature_x}_{feature_y}")


class ScoringAnalysis(Analyser):
    """
    The ScoringAnalyser class provides methods for analyzing the performance of
    a machine learning model.
    """

    def __init__(self, on):
        super().__init__(on)

    def confusion_matrix(self, in_percentage=False):
        """
        Compute the confusion matrix.

        Parameters
        ----------
        in_percentage : bool, optional
            Whether to return the confusion matrix in percentage, by default False

        Returns
        -------
        pandas.DataFrame
            The confusion matrix
        """

        matrix = np.array([[0, 0], [0, 0]])
        for i, (_, val) in enumerate(self.indexes):
            proba = self.estimators[i].predict_proba(self.X.iloc[val])[:, -1]
            y_pred = proba > self.threshold
            matrix += metrics.confusion_matrix(self.y.iloc[val], y_pred)
            matrix = matrix / len(self.indexes)
        columns = ['predicted : 0', 'predicted : 1']
        index = ['real : 0', 'real : 1']

        matrix = pd.DataFrame(matrix, index=index, columns=columns,
                              ).round(decimals=1)
        if in_percentage:
            matrix = matrix / matrix.sum().sum() * 100
            matrix = matrix.round(decimals=1)
        return matrix

    def __interpolate_roc(self, _):
        self._hidden_metrics["roc_curve_interp"] = utils.interpolate_roc(
            self.metrics["roc_curve"], __fpr_sampling__)

    def plot_roc_curve(
            self,
            plot_method="mean",
            plot_train: bool = False,
            c='C0',
            cmap: str = "inferno",
            label: str = "",
            mode: str = "std",
            label_iter: iter = None,
            plot_base: bool = True,
            **kwargs):
        """
        Plot the ROC curve.

        Parameters
        ----------

        plot_method : str,
            Select the type of plot for ROC curve

            - "beam" (default) to plot all the curves using shades
            - "all" to plot each ROC curve
            - "mean" plot the mean ROC curve

        plot_train: bool
            If True the train ROC curves will be plot, default False.

        c: str
            Not used only with plot_method="all". Set the color of ROC curve

        cmap: str

        label
        mode
        label_iter
        plot_base: bool,
            Plot basic ROC curve helper
        kwargs:
            Deprecated

        Returns
        -------

        """
        self._compute_metric("roc_curve", metrics.roc_curve)

        self.__interpolate_roc(self.indexes)
        roc = self._hidden_metrics["roc_curve_interp"]

        select_i = "train" if plot_train else "test"
        list_tpr = [roc[i][select_i][1] for i in range(len(self.indexes))]
        list_fpr = [roc[i][select_i][0] for i in range(len(self.indexes))]

        if plot_method not in ["beam", "all", "mean"]:
            raise ValueError(
                f"argument plot_method={plot_method} is not recognize")

        plot_all: bool = plot_method == "all"
        plot_beam: bool = plot_method == "beam"
        plot_mean: bool = plot_method == "mean"

        plotting.roc_plot_bundle(list_fpr, list_tpr, mean_fpr=__fpr_sampling__,
                                 plot_all=plot_all, plot_beam=plot_beam,
                                 label_iter=label_iter,
                                 cmap=cmap, plot_mean=plot_mean,
                                 c=c, mode=mode, label=label)

        if plot_base:
            plotting.roc_plot_base()

        logger.logger.log_artifact(plot.gcf(), "roc_curve")
        return plot

    def compute_threshold(
            self,
            method: str = "total_population",
            value: float = 0.5,
            metric: typing.Callable = None):
        """
        Compute threshold using various heuristics

        Parameters
        ----------
        method : str, optional
            The method to compute the threshold, by default "total_population"

            - total population : compute threshold so that the percentage of
            positive prediction is equal to `value`
            - fpr : compute threshold so that the false positive rate
            is equal to `value`
            - optimize_metric : compute threshold so that the metric is optimized
            `value` parameter is ignored, `metric` parameter must be provided

        value : float, optional
            The value to use for the threshold computation, by default 0.5
        metric : typing.Callable, optional
            The metric function to use for the threshold computation, by default None

        Returns
        -------
        float
            The computed threshold
        """
        th = []

        if method == "total_population":
            for i, _ in enumerate(self.indexes):
                th.append(np.percentile(self.predictions[i]["test"], value))
        elif method == "fpr":
            if "roc_curve" not in self.metrics.keys():
                self._compute_metric("roc_curve", metrics.roc_curve)
            self.__interpolate_roc(self.indexes)
            for i, _ in enumerate(self.indexes):
                roc = self._hidden_metrics["roc_curve_interp"][i]["test"]
                idx = np.argmin(np.abs(roc[0] - value))
                th.append(roc[2][idx])
        elif method == "optimize_metric":
            name = "threshold_criterion"
            if metric is None:
                raise ValueError("Argument metric must not be not None")
            self._hidden_metrics[name] = {}
            for i, (train, test) in enumerate(self.indexes):
                ths = np.unique(self.predictions[i]["test"])
                temp = {}
                for th_ in ths:
                    temp[th_] = metric(
                        self.y.iloc[test],
                        self.predictions[i]["test"] > th_)
                self._hidden_metrics[name][i] = dict(
                    test=pd.Series(temp).idxmax())
                th = [self._hidden_metrics[name][i]["test"] for i in
                      self._hidden_metrics[name].keys()]
        else:
            raise ValueError(f"method {method} is not recognized")
        self.__threshold = np.nanmean(th)
        return self.__threshold

    def plot_threshold(self, **plot_kwargs):
        """
        Plot the threshold on fpr/tpr axes

        Parameters
        ----------
        plot_kwargs : dict, optional
            Additional keyword arguments to pass to the scatter plot function

        Returns
        -------
        matplotlib.pyplot
            The threshold plot
        """
        fpr = []
        tpr = []
        for i, _ in enumerate(self.indexes):
            roc = self._hidden_metrics["roc_curve_interp"][i]["test"]
            idx = np.argmin(np.abs(roc[2] - self.__threshold))
            fpr.append(roc[0][idx])
            tpr.append(roc[1][idx])
        plot.scatter(fpr, tpr, **plot_kwargs)
        logger.logger.log_artifact(plot.gcf(), "roc_threshold")

    @property
    def threshold(self):
        return self.__threshold


class RegressionAnalysis(Analyser):
    """
    Analyser class for performing analysis on a regression model.

    Parameters
    ----------
    on : str
        The type of analysis to perform. Possible values are
        "indexes_train_test" or "indexes_val".

    Attributes
    ----------
    _hidden_metrics : dict
        Dictionary to store additional metrics that are not displayed.

    Methods
    -------
    variable_importance()
        Compute the feature importance for each estimator.
    compute_metrics(metric: dict)
        Compute the specified metrics for each estimator.
    get_train_metrics() -> pd.DataFrame
        Get the computed metrics for the training set.
    get_test_metrics() -> pd.DataFrame
        Get the computed metrics for the test set.
    plot_variable_importance(mode="minmax", color="darkblue", cmap="flare")
        Plot the variable importance.
    plot_prediction_versus_real
        Plot prediction versus real values
    plot_errors_pairgrid
        Plot pair grid errors

    Properties
    ----------
    metrics : dict
        The computed metrics.
    """

    def __init__(self, on):
        super().__init__(on)

    def compute_predictions_errors(self, fun=None):
        self.errors = {}
        if fun is None:
            def fun(yt_, yp_):
                return (yt_ - yp_) ** 2

        for i, (_, val) in enumerate(self.indexes):
            y_pred = self.predictions[i]["test"]
            y_true = self.y.iloc[val]
            self.errors[i] = fun(y_true, y_pred)
        return pd.DataFrame(self.errors)

    def plot_prediction_versus_real(self,
                                    colormap=plot.get_cmap("rainbow")):
        ax = plot.gca()
        values = np.array(self.y)
        for i, (_, val) in enumerate(self.indexes):
            c = colormap(i / (len(self.indexes) + 1))
            y_pred = self.predictions[i]["test"]
            ax.scatter(self.y.iloc[val], y_pred, color=c, zorder=2)
            values = np.concatenate((y_pred, values))
        mi, ma = values.min(), values.max()

        ax.plot([mi, ma], [mi, ma], ls="--", alpha=0.5, color="k", zorder=1)
        ax.set_aspect('equal', 'box')
        ax.grid()
        ax.set_ylabel("Predicted values")
        ax.set_xlabel("Real values")
        logger.logger.log_artifact(plot.gcf(), "true_vs_predicted")

    def plot_errors_pairgrid(self, fun=None, number_percentiles=4,
                             palette="rocket_r", features=None):

        import seaborn as sns
        from sklearn.preprocessing import KBinsDiscretizer
        if features is None:
            features = self.X.columns
        df_plot = self.X[features].copy()
        df_plot["error"] = self.compute_predictions_errors(fun=fun).mean(axis=1)
        df_plot = df_plot.dropna()
        df_plot["error quartile"] = (KBinsDiscretizer(
            n_bins=number_percentiles,
            encode="ordinal",
            strategy="quantile"
        ).fit_transform(df_plot[["error"]]).astype(int) + 1).astype(str)
        df_plot = df_plot.sort_values("error quartile")
        g = sns.PairGrid(df_plot, hue="error quartile", palette=palette,
                         x_vars=features, y_vars=features,
                         )
        g.map_diag(sns.histplot, multiple="stack")
        g.map_offdiag(sns.scatterplot, size=df_plot["error"])
        g.add_legend(title="", adjust_subtitles=True)
        logger.logger.log_artifact(plot.gcf(), "pair_grid")


class PermutationFeatureImportance(ModelComponent):
    """
    Class for doing permutation feature importance

    Parameters
    ----------
    n_repeat: int
        The number of times to permute a feature.
    random_state: int
        The pseudo-random number generator to control the permutations of each feature.
    n_job: int
        The number of jobs to run in parallel. If n_job = -1, it takes all processors.
    max_samples: int or float
        The number of samples to draw from X to compute feature importance in each repeat (without replacement).
        If int, then draw max_samples samples.
        If float, then draw max_samples * X.shape[0] samples.
    color: str
        The color for bar plot.

    Methods
    -------
    plot_permutation_feature_importance()
        Plotting the result of feature permutation ONLY on the TRAINING SET
    """

    def __init__(self, n_repeat: int = 5, random_state: int = 42, n_job: int = 2,
                 scoring: str = None, max_samples: typing.Union[int, float] = 0.7, color: str = 'darkblue'):
        self.n_repeat = n_repeat
        self.random_state = random_state
        self.n_job = n_job
        self.scoring = scoring
        self.max_samples = max_samples
        self.color = color

    def __call__(self, project: "Project", model: "ModelEvaluation"):
        self.indexes = project.validation_strategy.indexes_train_test
        self.estimator = model.unfit_estimator
        train_id = self.indexes[0][0]  # get only training indices
        self.X_train = project.X.iloc[train_id]
        self.y_train = project.y.iloc[train_id]
        self.plot_permutation_feature_importance()

    def plot_permutation_feature_importance(self):

        args = dict(edgecolor='white', color=self.color)
        ax = plot.gca()
        train_res = permutation_importance(self.estimator, self.X_train, self.y_train, n_repeats=self.n_repeat,
                                           random_state=self.random_state, n_jobs=self.n_job, scoring=self.scoring,
                                           max_samples=self.max_samples)

        sorted_importances_idx = train_res.importances_mean.argsort()
        ax.barh(range(len(sorted_importances_idx)), train_res.importances_mean[sorted_importances_idx],
                align='center', **args)
        ax.set_yticks(range(len(sorted_importances_idx)), np.array(self.X_train.columns)[sorted_importances_idx])
        ax.set_title('Train set')
        plot.tight_layout()
        logger.logger.log_artifact(plot.gcf(), "permutation_feature_importance")
