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

import pandas as pd

from palma.components.base import ProjectComponent
from palma.utils import utils


class ValidationStrategyChecker(ProjectComponent):
    def __init__(self, ):
        pass

    def __call__(self, project: "Project"):
        indexes_test = project.validation_strategy.indexes_train_test
        indexes_val = project.validation_strategy.indexes_val

        df_val = utils.get_splitting_matrix(project.X, indexes_val, expand=True)
        df_tra = utils.get_splitting_matrix(project.X, indexes_test,
                                            expand=True)
        df_tra.columns = ["test_" + str(c) for c in df_tra.columns]
        df = pd.concat((df_val, df_tra), axis=1)

        self.check_no_leakage_in_test(df, df_tra.columns)
        self.check_no_leakage_in_validation(df_val)

    @staticmethod
    def check_no_leakage_in_test(df, test_columns):
        for i, c in enumerate(test_columns):
            if i % 2 == 1:

                assert df[df[c]].sum().sum() == df[df[c]].__len__()

    @staticmethod
    def check_no_leakage_in_validation(df):
        for i, c in enumerate(df.columns):
            if i % 2 == 1:
                assert sum(df.loc[:, c] & df.loc[:, df.columns[i - 1]]) == 0


