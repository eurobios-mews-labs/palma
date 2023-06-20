# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from hashlib import blake2b
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.model_selection._split import BaseShuffleSplit, BaseCrossValidator


def _index_to_bool(array, length):
    ret = np.zeros(length) * False
    ret[array] = True
    return ret.astype(bool)


def _bool_to_index(array):
    return np.where(array)[0]


class ValidationStrategy(object):

    def __init__(
            self,
            splitter: Union[
                BaseShuffleSplit,
                BaseCrossValidator,
                List[tuple],
                List[str]],
            **kwargs
    ) -> None:

        self.__splitter = splitter
        if hasattr(self.__splitter, "split"):
            self._splitter_args = self.__splitter.__dict__
        else:
            self._splitter_args = ""
        if hasattr(self.__splitter, "random_state"):
            if getattr(self.__splitter, "random_state") is None:
                self.__splitter.__setattr__("random_state", np.random.choice(
                    100000
                ))

    def __call__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_test: pd.DataFrame = None,
            y_test: pd.Series = None,
            groups=None,
            **kwargs
    ):

        if X_test is not None and y_test is not None:
            n_train, n_test = len(X), len(X_test)
            X = pd.concat((X, X_test), axis=0)
            y = pd.concat((y, y_test), axis=0)

            self._train_index = np.array(range(n_train))
            self._test_index = np.array(range(n_train, n_train + n_test))
        else:
            if hasattr(self.__splitter, "split"):
                train, test = list(self.__splitter.split(X, y, groups))[-1]
                self._train_index = train
                self._test_index = test
        if hasattr(self.__splitter, "split"):
            if groups is not None:
                groups_ = np.array(groups)[self._train_index]
            else:
                groups_ = groups
            self.__indexes = list(self.__splitter.split(
                X.iloc[self._train_index],
                y.iloc[self._train_index],
                groups_))
        self.__indexes = self.__correct_nested(X)
        self.__id = blake2b(
            str(self).encode('utf-8'), digest_size=5).hexdigest()

        return X, y

    def __correct_nested(self, X):
        ret = []
        tr = _index_to_bool(self._train_index, len(X))
        for train, test in self.__indexes:
            r = []
            for t in (train, test):
                trb = _index_to_bool(t, len(self._train_index))
                b_train = tr.copy()
                b_train[b_train] = trb
                r.append(_bool_to_index(b_train))
            ret.append(tuple(r))
        return ret

    def __str__(self) -> str:
        splitting_strategies = 'train/test split with args: {}' \
            .format(self._splitter_args)
        return splitting_strategies

    @property
    def test_index(self) -> np.ndarray:
        return self._test_index

    @property
    def train_index(self) -> np.ndarray:
        return self._train_index

    @property
    def indexes_val(self) -> list:
        return self.__indexes

    @property
    def indexes_train_test(self) -> list:
        return [(self._train_index, self._test_index)]

    @property
    def id(self):
        return self.__id

    @property
    def splitter(self):
        return self.__splitter
