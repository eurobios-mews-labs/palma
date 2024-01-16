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

from palma.base.project import Project


class Component(object, metaclass=ABCMeta):

    def __str__(self):
        return self.__class__.__name__


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
