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

from datetime import datetime
from typing import Union, Dict

from palma.base import engine as eng
from palma.utils.utils import get_hash
import logging


class ModelSelector:
    """
    Wrapper to optimizers selecting the best model for a Project.

    The optimization can be launched with the ``start`` method.
    Once the optimization is done, the best model can be accessed as the ``best_model_`` attribute.

    Parameters
    ----------
    - engine (str): Currently accepted values are "FlamlOptimizer" or
      "AutoSklearnOptimizer" (the latter is deprecatted).
    - engine_parameters (dict): parameters passed to the engine.

    Methods
    -------
    - start(project: Project): look for best model


    """

    def __init__(
            self,
            engine: Union[str, eng.BaseOptimizer],
            engine_parameters: Dict
    ) -> None:

        self.__date = datetime.now()
        self.__run_id = get_hash(date=self.__date)
        self.__parameters = engine_parameters
        if engine == "AutoSklearnOptimizer":
            self.engine = eng.AutoSklearnOptimizer
        elif engine == "FlamlOptimizer":
            self.engine = eng.FlamlOptimizer
        else:
            raise ValueError(f"Optimizer {engine} not implemented")

    def start(self, project: "Project"):
        from palma import logger
        self.engine_ = self.engine(
            project.problem,
            engine_parameters=self.__parameters,

        )

        logging.disable()
        self.engine_.optimize(
            project.X.iloc[project.validation_strategy.train_index],
            project.y.iloc[project.validation_strategy.train_index],
            splitter=project.validation_strategy,
        )
        self.best_model_ = self.engine_.estimator_
        logging.basicConfig(level=logging.DEBUG)

        logger.logger.log_artifact(
            self.engine_.estimator_,
            self.__run_id)
        try:
            logger.logger.log_metrics(
                {"best_estimator": str(self.best_model_)}, 'model_selection'
            )
        except:
            pass

    @property
    def run_id(self) -> str:
        return self.__run_id
