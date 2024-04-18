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

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors


def plot_correlation(df: pd.DataFrame, cmap: str = 'RdBu_r',
                     method: str = "spearman", linewidths=1, fmt="0.2f",
                     vmin=-1, vmax=1):
    corr = df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=linewidths, fmt=fmt)
    plot.tight_layout()


def plot_splitting_strategy(X: pd.DataFrame, y: pd.Series,
                            iter_cross_validation: iter,
                            cmap, sort_by=None, modulus=1):
    from palma.utils.utils import get_splitting_matrix
    data = get_splitting_matrix(X, iter_cross_validation)
    y_ = pd.Series(y)
    if sort_by == "cv":
        data = data.sort_values(by=data.columns.to_list())
        y_ = y_.loc[data.index]
    elif sort_by == "target":
        y_ = y_.sort_values()
        data = data.loc[y_.index]
    elif sort_by in data.columns:
        indexes = X.sort_values(by=sort_by).index
        y_ = y_.loc[indexes]
        data = data.loc[indexes]
    select = np.arange(0, len(data), step=modulus)
    fig, ax = plot.subplots(figsize=(12, 0.6 * (len(data.columns) + 1)),
                            nrows=len(data.columns) + 1, sharex=True)

    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    cmap = plot.cm.Blues

    for i, c in enumerate(data.columns):
        sns.heatmap(data[[c]].T.iloc[:, select], cmap=cmap, cbar_ax=cbar_ax,
                    vmin=0,
                    ax=ax[i])
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    mappable = plot.cm.ScalarMappable(norm=norm, cmap=cmap)

    mappable.set_array([])
    mappable.set_clim(-0.5, 3 + 0.5)
    color_bar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
    color_bar.set_ticks(np.linspace(0.5, 2.5, 3))
    color_bar.set_ticklabels(["Out", "Train", "Test"])

    target = pd.DataFrame(y_.values, columns=["Target \n values"])
    sns.heatmap(target.T.iloc[:, select], cmap="magma", ax=ax[-1], cbar=False)

    return data


def plot_variable_importance(
        variable_importance: pd.DataFrame,
        mode="minmax",
        color="darkblue",
        cmap="flare",
        alpha=0.2, **kwargs
):
    variable_importance = variable_importance.copy()
    variable_importance.index = variable_importance.index.astype(str)
    m = np.array(variable_importance.mean(axis=1)).ravel()
    variable_importance["mean"] = m

    variable_importance = variable_importance.sort_values(
        by="mean", ascending=mode == "boxplot")
    variable_importance = variable_importance.drop(columns=["mean"])

    if mode == "minmax":
        m_ = np.array(variable_importance.min(axis=1)).ravel()
        M_ = np.array(variable_importance.max(axis=1)).ravel()
        upper = M_
        lower = m_
        label = ("min - mean", "[min, max]")
    elif mode == "std":
        std = np.array(variable_importance.std(axis=1)).ravel()
        m_ = variable_importance.mean(axis=1).ravel()
        lower = m_ - std
        upper = m_ + std
        label = ("mean - std", "mean +/- std")

    elif mode == "boxplot":
        import seaborn as sns
        sns.boxplot(data=variable_importance.T, orient="h", palette=cmap)
        ax = plot.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        return variable_importance
    else:
        raise ValueError(f"Mode {mode} is unknown")

    y_outer = np.where(np.abs(lower) > np.abs(upper), lower, upper)
    y_inner = np.where(np.abs(upper) > np.abs(lower), lower, upper)
    index = variable_importance.index.to_numpy()

    args = dict(edgecolor='white', color=color)
    ax = plot.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plot.barh(index, y_inner, alpha=alpha,
              left=np.zeros_like(y_inner), **args)
    plot.barh(index, y_outer - y_inner,
              left=y_inner,
              label=label[1], **args)
    if variable_importance.shape[1] > 1:
        plot.legend()
    plot.tight_layout()
    return variable_importance


def roc_plot_bundle(list_fpr, list_tpr,
                    mean_fpr=np.linspace(0, 1, 100),
                    plot_all=False,
                    plot_beam=True,
                    cmap="inferno",
                    plot_mean=True,
                    c="b",
                    label_iter=None,
                    mode="std",
                    label="",
                    **args):
    from sklearn.metrics import auc
    from matplotlib import cm
    ax = plot.gca()
    tpr_interp_list, auc_list = [], []
    for i in range(len(list_tpr)):
        tpr, fpr = list_tpr[i], list_fpr[i]
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        if plot_all:
            cm_ = cm.get_cmap(cmap, 12) if cmap is not None else None
            c = cm_(i / len(list_tpr)) if cmap is not None else c
            if label_iter is None:
                label = 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc)
            else:
                label = label_iter[i]
            ax.plot(fpr, tpr, lw=2, alpha=0.8, color=c,
                    label=label)
        tpr_interp_list.append(tpr)
        tpr_interp_list[-1][0] = 0.0

    if plot_mean:
        mean_tpr = np.mean(tpr_interp_list, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.round(auc(mean_fpr, mean_tpr), 2)
        std_auc = np.round(np.std(np.array(auc_list)), 2)
        label_ = f'ROC curve (AUC = {mean_auc} $\pm$ {std_auc})'
        ax.plot(mean_fpr, mean_tpr, color=c,
                label=label + ' ' + label_,
                lw=2, alpha=.8)

    if mode == "minmax":
        tpr_upper = np.max(tpr_interp_list, axis=0)
        tpr_lower = np.min(tpr_interp_list, axis=0)
    elif mode == "std":
        mean_ = np.mean(tpr_interp_list, axis=0)
        std_ = np.std(tpr_interp_list, axis=0)
        tpr_upper = mean_ + std_
        tpr_lower = mean_ - std_
    else:
        raise ValueError(f"Argument mode {mode} is unknown")

    if plot_beam:
        ax.fill_between(mean_fpr, tpr_lower,
                        tpr_upper, color=c, alpha=0.1)


def roc_plot_base():
    ax = plot.gca()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.6)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.axis("equal")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
