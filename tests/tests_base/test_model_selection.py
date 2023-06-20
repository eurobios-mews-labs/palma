# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import pytest

from palma.base.model_selection import ModelSelector


@pytest.fixture(scope="module")
def get_model_selector_flaml(regression_data):
    engine_parameters = dict(time_budget=10, task='regression')
    return ModelSelector(
        engine='FlamlOptimizer',
        engine_parameters=engine_parameters,
    )


def test_model_selector_flaml_engine(get_model_selector_flaml):
    assert hasattr(get_model_selector_flaml, 'engine'), \
        'Run instance should have a engine attribute'


def test_run_classification_run_id(get_model_selector_flaml):
    assert hasattr(get_model_selector_flaml, 'run_id'), \
        'Model selector instance should have a run_id attribute'


def test_run_unknown_engine():
    engine_parameters = {
        'time_left_for_this_task': 10
    }
    with pytest.raises(ValueError) as exc_info:
        ModelSelector(
            engine='UnknownOptimizer',
            engine_parameters=engine_parameters)
    assert type(exc_info.value) == ValueError, 'Selected Optimizer does not exist'
