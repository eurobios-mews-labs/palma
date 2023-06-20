# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from palma.preprocessing.pca import PCA
import pytest


@pytest.fixture(scope="module")
def get_pca(learning_data):
    project, learn, X, y = learning_data
    pca = PCA(X.iloc[:40, :10])
    return pca, X.iloc[:40, :10]


def test_pca_transform(get_pca):
    pca, X = get_pca
    assert pca.transform(X).shape[1] == pca.nb_component
    pca.set_nb_components(n=2)
    assert pca.nb_component == 2


def test_all_plot(get_pca):
    import matplotlib
    matplotlib.use("agg")
    pca, X = get_pca
    pca.set_nb_components(variance_threshold=0.8)
    pca.get_variables_contributions()
    pca.get_correlation()
    pca.get_individual_contributions()

    pca.plot_eigen_values()
    pca.plot_cumulated_variance(color="darkblue")
    pca.plot_circle_corr()
    pca.plot_correlation_matrix()
    pca.plot_factorial_plan(X)
    pca.plot_var_cp(X, n_col=2, figsize=(15, 10))
    pca.plot_variance_bar()
