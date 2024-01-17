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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, data: pd.DataFrame,
                 prefix_name="pc") -> None:
        """

        Parameters
        ----------
        data :
            input data frame
        prefix_name :
            the prefix to be used in the component name
        """
        self.data_train: pd.DataFrame = data.copy()
        self.index = self.data_train.index
        self.n = data.shape[0]
        self.p = data.shape[1]

        # fit PCA and get explained variance
        self.sc = StandardScaler()
        self.scaled_data = self.sc.fit_transform(self.data_train)
        self.pca = SKPCA(svd_solver='auto', random_state=0)
        self.pca.fit(self.scaled_data)
        self.explained_variance = (self.n - 1) / self.n * self.pca.explained_variance_
        self.component_names = [f"{prefix_name}{i}" for i in range(1, self.data_train.shape[1] + 1)]
        self.eigen_values = (self.n - 1) / self.n * self.pca.explained_variance_
        self.__nb_comp = self.data_train.shape[1]

    def set_nb_components(self, n=None, variance_threshold: float = None, **kwargs):
        if n is not None:
            self.__nb_comp = n
            return self.__nb_comp
        if variance_threshold is not None:
            self.__nb_comp = np.where(np.cumsum(self.pca.explained_variance_ratio_) >= variance_threshold)[0][
                0] + 1
            self.__nb_comp = min(self.data_train.shape[1], self.__nb_comp)
            return self.__nb_comp

    @property
    def nb_component(self):
        return self.__nb_comp

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ret = pd.DataFrame(
            self.pca.transform(self.sc.transform(X))[:, :self.nb_component],
            index=X.index,
            columns=self.component_names[:self.nb_component])
        return ret

    def __get_corr(self, n_components=None) -> np.ndarray:
        if n_components is None:
            n_components = self.nb_component

        sqrt_eigen_values = np.sqrt(self.eigen_values)
        # correlation between variables and axes
        correlation = np.zeros((self.p, self.p))
        for k in range(n_components):
            correlation[:, k] = self.pca.components_[k, :] * sqrt_eigen_values[k]
        return correlation

    def get_correlation(self, n_components=None) -> pd.DataFrame:
        if n_components is None:
            n_components = self.nb_component
        correlation = self.__get_corr(n_components)
        corr = pd.DataFrame(correlation[:, :len(self.component_names)], index=self.data_train.columns,
                            columns=self.component_names)
        corr.index.name = "Features"
        return corr

    def get_variables_contributions(self, n_components=None) -> pd.DataFrame:
        if n_components is None:
            n_components = self.nb_component
        correlation = self.__get_corr(n_components)
        cos2_var = correlation ** 2

        contributions = cos2_var.copy()
        for k in range(n_components):
            contributions[:, k] = cos2_var[:, k] / self.eigen_values[k]
        ctr_var = pd.DataFrame({'Features': self.data_train.columns})
        for i in range(n_components):
            ctr_var["CTR" + str(i + 1)] = contributions[:, i]
        ctr_var = ctr_var.set_index("Features")
        return ctr_var

    def get_individual_contributions(self, n_components=None) -> pd.DataFrame:
        if n_components is None:
            n_components = self.nb_component

        data_transformed = self.transform(self.data_train)
        ctr_ind = data_transformed ** 2
        for j in range(n_components):
            colnames = ctr_ind.columns
            ctr_ind[colnames[j]] = ctr_ind[colnames[j]] / (self.n * self.eigen_values[j])
        return ctr_ind

    def plot_eigen_values(self) -> None:
        plt.plot(np.arange(1, self.p + 1), self.eigen_values, marker=".")
        plt.ylabel("Eigen values")
        plt.xlabel("Factor number")
        plt.grid()
        plt.show()

    def plot_cumulated_variance(self, color="tab:blue") -> None:
        cumulated_variance_ratio = np.concatenate(([0], np.cumsum(self.pca.explained_variance_ratio_)))
        plt.plot(np.arange(0, self.p + 1)[:self.nb_component + 1], cumulated_variance_ratio[:self.nb_component + 1],
                 marker=".",
                 color=color)
        plt.plot(np.arange(0, self.p + 1)[self.nb_component:], cumulated_variance_ratio[self.nb_component:], marker=".",
                 color=color, ls="--")
        plt.axvline(self.nb_component, color='red')
        plt.ylabel("Cumulated proportion of explained variance")
        plt.xlabel("Number of components")
        plt.show()

    def plot_circle_corr(self) -> None:

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        corr = self.get_correlation()

        for j in range(self.p):
            plt.annotate(
                self.data_train.columns[j],
                (corr.iloc[j, 0], corr.iloc[j, 1]))

        plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

        circle = plt.Circle((0, 0), 1, color='tab:blue', fill=False)
        plt.gca().add_artist(circle)
        plt.show()

    def plot_correlation_matrix(self) -> None:
        import seaborn as sns
        data_plot = self.get_correlation()

        sns.heatmap(data_plot, annot=True, fmt=".2f", cmap='RdBu_r',
                    vmin=-1, vmax=1, cbar_kws={"shrink": .8}, linecolor="w",
                    linewidths=.5
                    )

    def plot_factorial_plan(self, X: pd.DataFrame, x_axis="pc1", y_axis="pc2", c=None, cmap=None) -> None:
        x = self.transform(X)
        plt.xlim(x[x_axis].min() - 1,
                 x[x_axis].max() + 1)
        plt.ylim(x[y_axis].min() - 1,
                 x[y_axis].max() + 1)
        # placing labels
        for i in range(len(x)):
            plt.annotate(x.index[i], (
                x[x_axis].iloc[i], x[y_axis].iloc[i]))
        # Add axes
        plt.plot([-8, 8], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-8, 8], color='silver', linestyle='-', linewidth=1)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()

    def plot_var_cp(self, X: pd.DataFrame, n_col=3, figsize=(10, 10), x_axis="pc1", y_axis="pc2") -> None:

        i = 0
        fig, ax = plt.subplots(nrows=self.p // n_col + int(self.p % n_col != 0),
                               ncols=n_col,
                               figsize=figsize, sharex=True, sharey=True,
                               )
        x = self.transform(X)
        for i, c in enumerate(X.columns):
            ax[i // n_col, i % n_col].scatter(x[x_axis],
                                              x[y_axis],
                                              c=X[c],
                                              cmap="rainbow")
            ax[i // n_col, i % n_col].annotate(c, (x[x_axis].min(), x[y_axis].max()),
                                               bbox=dict(boxstyle="round", fc="w"))
        for j in range(i + 1, (self.p // n_col + int(self.p % n_col != 0)) * n_col):
            ax[j // n_col, j % n_col].axis('off')

        plt.show()

    def plot_variance_bar(self, separator=0.5) -> None:
        evr = np.concatenate(([0], np.cumsum(self.pca.explained_variance_ratio_))) * 100
        # evr = evr.astype(int)
        d = separator
        tuple_bar = [(evr[i], evr[i + 1] - evr[i] - d) for i in range(evr.__len__() - 1)]
        tuple_bar = [(c[0], c[1]) for c in tuple_bar if c[1] > 0]
        c = [plt.get_cmap('rainbow_r')(i / len(tuple_bar)) for i in range(len(tuple_bar))]

        fig, ax = plt.subplots(dpi=300, figsize=(5, 2))
        ax.broken_barh(tuple_bar, [0, 0.5], facecolors=c)
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 100)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_yticks([-1, 0.25])
        ax.set_xticklabels([f"{int(t)}%" for t in ax.get_xticks()])

        ax.set_axisbelow(True)

        ax.set_yticklabels(['', ""])
        ax.grid(axis='x')

        for i, x in enumerate(tuple_bar):
            if x[1] < 5:
                break
            ax.text(x[0] + 0.5, 0.2, f"{int(x[1] + d)}%", fontsize=8)
        for i, x in enumerate(tuple_bar):
            if i == len(self.component_names) or x[1] < 5:
                break
            up = 0.11 * np.mod(i, 2) if len(tuple_bar) > 10 else 0
            ax.text(x[0], 0.51 + up, self.component_names[i], fontsize=8, color=c[i],
                    fontweight="bold")
        var_explained = np.sum(self.pca.explained_variance_ratio_[:self.nb_component]) * 100
        ax.broken_barh([(0, var_explained), (var_explained + d, 100)], [-0.4, -0.2], facecolor=["k", "gray"])
        ax.text(var_explained / 2 - 18, -0.55, f"Explained variance : {int(var_explained)}%", fontsize=8, color="w")

        ax.broken_barh([(65, 100), (0, 0)], [-0.63, -0.3], facecolor=["w", "w"])
        ax.text(65, -0.70, f"Remaining variance {100 - int(var_explained)}%", fontsize=8, color="gray")


if __name__ == '__main__':
    pass
    import matplotlib

    matplotlib.use("qt5agg")
    import pandas as pd
    import seaborn as sns
    from sklearn.datasets import make_classification
    from palma import Project
    from palma import components

    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)
    from palma.utils.plotting import plot_correlation

    plt.figure(figsize=(6, 6))

    plot_correlation(X.iloc[:, :8], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    project = Project(problem="classification", project_name="test")
    # project.add(components.FileSystemLogger("/tmp/"))
    pca: PCA = PCA(X)
    pca.get_variables_contributions()
    pca.get_correlation()
    pca.get_individual_contributions()

    plt.figure(figsize=(6, 6), dpi=200)
    pca.plot_eigen_values()
    plt.savefig("../pca_eigen_values.png")
    #
    plt.figure(figsize=(6, 6), dpi=200)
    pca.plot_cumulated_variance(color="darkblue")
    plt.savefig("../pca_cumul_variance.png")

    plt.figure(figsize=(6, 6), dpi=200)
    pca.plot_circle_corr()
    plt.savefig("../circles.png")

    plt.figure(figsize=(20, 6), dpi=200)
    pca.plot_correlation_matrix()
    plt.savefig("../corr_matrix.png")

    plt.figure(figsize=(6, 6), dpi=200)
    pca.plot_factorial_plan(X)
    plt.savefig("../fact_plan.png")

    pca.plot_var_cp(X, n_col=4, figsize=(15, 10))
    plt.savefig("../real_plan.png")

    pca.plot_variance_bar()
