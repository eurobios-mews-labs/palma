# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import os
import tempfile

import pytest


class TestBadComponent:
    pass


def test_date_property(classification_project):
    assert classification_project.date, "Missing date attribute"


def test_project_name_property(classification_project):
    assert classification_project.project_name, "Missing name attribute"


def test_is_built_property(classification_project):
    assert classification_project.is_started, "bad is_started value"


def test_problem_property(classification_project):
    assert classification_project.problem == "classification", "bad pb type"


def test_study_name_property(classification_project):
    assert "_" in classification_project.study_name, "bad use case name"


def test_add_logger(classification_project):
    project = classification_project
    print(project.components)
    assert hasattr(project, "_logger"), "Logger is not properly set"


def test_add_bad_component(unbuilt_classification_project):
    project = unbuilt_classification_project
    with pytest.raises(TypeError) as exc_info:
        project.add(TestBadComponent())
    assert type(exc_info.value) == TypeError, "Wrong error for bad component"


def test_project_id(classification_project):
    assert classification_project.project_id, "Missing project\
        id attribute"


def test_project_is_started(classification_project):
    assert classification_project.is_started, "bad is_started \
    value"


def test_project_splitting_strategy_property(
        classification_project
):
    assert hasattr(
        classification_project,
        "validation_strategy"
    ), "Mssing validation_strategy attribute"


def test_project_X_property(classification_project):
    assert hasattr(
        classification_project,
        "X"
    ), "Missing X attribute"


def test_project_y_property(classification_project):
    assert hasattr(
        classification_project,
        "y"
    ), "Missing y attribute"


def test_log_project(classification_project):
    path_to_check = os.path.join(
        tempfile.gettempdir(),
        classification_project.project_name,
        classification_project.project_name,
        "project.pkl"
    )
    assert os.path.exists(path_to_check), f"{path_to_check} doesn't exist"
