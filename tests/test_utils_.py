# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from palma.utils import plotting, utils


def test_plotting_corr(classification_data):
    plt.ioff()
    plotting.plot_correlation(pd.DataFrame(classification_data[0]).sample(20))


def test_utils_sha_dataframe(classification_data):
    plt.ioff()
    from sklearn.datasets import make_classification

    data, _ = make_classification(random_state=0)
    assert utils.hash_dataframe(
        pd.DataFrame(data[0]).sample(20, random_state=0),
        how="whole") == "2789de8" \
                        "6357239b2df" \
                        "7e7493f528707c"
    utils.hash_dataframe(pd.DataFrame(data[0]).sample(20), how="row_wise")
    utils.hash_dataframe(pd.DataFrame(data[0]).sample(20), how="types")
    with pytest.raises(TypeError):
        utils.hash_dataframe(pd.DataFrame(data[0]).sample(20), how="unknonw")


def test_utils_sha():
    assert utils.get_hash(
        t=1) == '6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b'


def test_get_splitting_matrix(learning_data):
    project, learn, X, y = learning_data
    df = utils.get_splitting_matrix(X, project.validation_strategy.indexes_val)
    assert len(df) == len(X)
