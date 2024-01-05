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

from abc import ABCMeta, abstractmethod
from typing import Union

from palma.base.model_selection import ModelSelector
from palma.base.project import Project


class Component(object, metaclass=ABCMeta):

    def __str__(self):
        return self.__class__.__name__

    def add_loger(self, project):
        self.__logger = project._logger
        return self

    @property
    def logger(self):
        return self.__logger


class ProjectComponent(Component):
    """
    Base Project Component class

    This object ensures that all subclasses Project component implements a
    """

    @abstractmethod
    def __call__(self, project: Project) -> None:
        ...


class ModelComponent(Component):
    """
    Base Model Component class
    """

    @abstractmethod
    def __call__(self, project: Project, model):
        pass


class Logger(metaclass=ABCMeta):
    """
    Logger is an abstract class that defines a common
    interface for a set of Logger-subclasses.

    It provides common methods for all possible subclasses, making it 
    possible for a user to create a custom subclass compatible  with 
    the rest of the components. 
    """

    def __init__(self, uri: str, **kwargs) -> None:
        self.__uri = uri

    def __call__(self, obj: Union["Project", "ModelSelector"]) -> None:
        if isinstance(obj, Project):
            self.log_project(obj)

        elif isinstance(obj, ModelSelector):
            self.log_run(obj)

    @abstractmethod
    def log_project(self, project: 'Project') -> None:
        ...

    @abstractmethod
    def _log_metrics(self, **kwargs) -> None:

        ...

    @abstractmethod
    def _log_params(self, **kwargs) -> None:

        ...

    @abstractmethod
    def _log_model(self, **kwargs) -> None:
        ...

    @property
    def uri(self):
        return self.__uri
