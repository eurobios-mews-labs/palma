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

from palma.utils import utils


def test_utils_sha():
    assert utils.get_hash(
        t=1) == ('6b86b273ff34fce19d6b804eff5a3f5747a'
                 'da4eaa22f1d49c01e52ddb7875b4b')


def test_get_splitting_matrix(learning_data):
    project, learn, X, y = learning_data
    df = utils.get_splitting_matrix(X, project.validation_strategy.indexes_val)
    assert len(df) == len(X)
