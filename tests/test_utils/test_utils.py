# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from palma.base.project import Project
from palma.utils import plotting, utils
from palma.utils.utils import get_hash, check_started


def test_plotting_correlation(classification_data):
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


def test_plot_splitting_strategy(learning_data):
    project, learn, X, y = learning_data
    plotting.plot_splitting_strategy(X, y,
                                     project.validation_strategy.indexes_val,
                                     sort_by="cv", cmap="rainbow")
    plotting.plot_splitting_strategy(X, y,
                                     project.validation_strategy.indexes_val,
                                     sort_by="target", cmap="rainbow")
    plotting.plot_splitting_strategy(X, y,
                                     project.validation_strategy.indexes_val,
                                     sort_by=X.columns[0], cmap="rainbow")
    utils.check_splitting_strategy(X, project.validation_strategy.indexes_val)


def test_get_splitting_matrix(learning_data):
    project, learn, X, y = learning_data
    df = utils.get_splitting_matrix(X, project.validation_strategy.indexes_val)
    assert len(df) == len(X)


kwargs = {
    'hp_space':
        {
            'n_estimators': [20, 30],
            'max_features': ['auto', 'log2'],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8, 10]
        },
    'date':
        datetime(2021, 3, 23, 10, 57, 54, 72210)
}


def test_get_hash():
    assert get_hash(
        **kwargs) == '7d673aea3d6ea639262a77e123111a03ea0dac7c3ed05d047a3a98d42e640831', \
        'Incorrect hash values'


def test_check_build_decorator():
    @check_started("test message", True)
    def test_func(project: Project):
        pass

    project_test = Project(
        project_name='project_name_test',
        problem='classification'
    )

    with pytest.raises(AttributeError) as exc_info:
        test_func(project_test)
    assert type(exc_info.value) == AttributeError, "AttributeError \
    should be raised"


def test_average_estimator(learning_data):
    project, learn, X, y = learning_data
    unique = np.unique(learn.avg_estimator_val_.predict(X))
    assert len(unique) <= 10
    assert max(np.unique(learn.avg_estimator_val_.predict_proba(X)[:, 1])) <= 1
