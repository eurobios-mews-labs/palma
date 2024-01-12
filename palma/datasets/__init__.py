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

import pandas as pd
from uci_dataset import load_credit_approval as load_credit_approval_uci


def load_credit_approval() -> typing.Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the Credit Approval dataset and prepares it for machine learning.

    The Credit Approval dataset is loaded using the uci_dataset library.
    The target variable 'A16' is transformed into binary labels, where '+'
    is encoded as True and '-' as False.
    Categorical features ('A13', 'A4', 'A6', 'A7', 'A5')
    are ordinal encoded using sklearn's OrdinalEncoder.
    Binary features ('A12', 'A9', 'A10', 'A1') are converted to boolean values.

       Returns
       -------
       Tuple[pd.DataFrame, pd.Series]
           A tuple containing the features (X) and target labels (y) for
            machine learning tasks.
       """
    from sklearn.preprocessing import OrdinalEncoder
    data = load_credit_approval_uci()
    X = data.drop(columns="A16")
    cat = ["A13", "A4", "A6", "A7", "A5"]
    y = data["A16"] == "+"
    X["A12"] = X["A12"] == "t"
    X["A9"] = X["A9"] == "t"
    X["A10"] = X["A10"] == "t"
    X["A1"] = X["A1"] == "a"
    X[cat] = OrdinalEncoder().fit_transform(X[cat])
    return X, y

