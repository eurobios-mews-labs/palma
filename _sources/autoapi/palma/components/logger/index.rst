:py:mod:`palma.components.logger`
=================================

.. py:module:: palma.components.logger


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.components.logger.Logger
   palma.components.logger.DummyLogger
   palma.components.logger.FileSystemLogger
   palma.components.logger.MLFlowLogger
   palma.components.logger._Logger




Attributes
~~~~~~~~~~

.. autoapisummary::

   palma.components.logger.mlflow
   palma.components.logger._logger


.. py:data:: mlflow

   

.. py:data:: _logger

   

.. py:class:: Logger(uri: str, **kwargs)


   
   Logger is an abstract class that defines a common
   interface for a set of Logger-subclasses.

   It provides common methods for all possible subclasses, making it
   possible for a user to create a custom subclass compatible  with
   the rest of the components.















   ..
       !! processed by numpydoc !!
   .. py:property:: uri


   .. py:method:: log_project(project: palma.base.project.Project) -> None
      :abstractmethod:


   .. py:method:: log_metrics(metrics: dict, path: str) -> None
      :abstractmethod:


   .. py:method:: log_params(**kwargs) -> None
      :abstractmethod:


   .. py:method:: log_model(**kwargs) -> None
      :abstractmethod:



.. py:class:: DummyLogger(uri: str, **kwargs)


   Bases: :py:obj:`Logger`

   
   Logger is an abstract class that defines a common
   interface for a set of Logger-subclasses.

   It provides common methods for all possible subclasses, making it
   possible for a user to create a custom subclass compatible  with
   the rest of the components.















   ..
       !! processed by numpydoc !!
   .. py:method:: log_project(project: palma.base.project.Project) -> None


   .. py:method:: log_metrics(metrics: dict, path: str) -> None


   .. py:method:: log_params(parameters: dict, path: str) -> None


   .. py:method:: log_model(estimator, path: str) -> None



.. py:class:: FileSystemLogger(uri: str = tempfile.gettempdir(), **kwargs)


   Bases: :py:obj:`Logger`

   



   :Parameters:

       **uri** : str
           root path or directory, from which will be saved artifacts and metadata 














   ..
       !! processed by numpydoc !!
   .. py:method:: log_project(project: palma.base.project.Project) -> None

      
      log_project performs the first level of backup as described
      in the object description. 

      This method creates the needed folders and saves an instance of         :class:`~palma.Project`.

      :Parameters:

          **project: :class:`~palma.Project`**
              an instance of Project














      ..
          !! processed by numpydoc !!

   .. py:method:: log_metrics(metrics: dict, path: str) -> None


   .. py:method:: log_model(estimator, path: str) -> None


   .. py:method:: log_params(parameters: dict, path: str) -> None



.. py:class:: MLFlowLogger(uri: str)


   Bases: :py:obj:`Logger`

   
   MLFlowLogger class for logging experiments using MLflow.


   :Parameters:

       **- uri (str): The URI for the MLflow tracking server.**
           ..





   :Raises:

       ImportError: If mlflow is not installed.
           ..







   :Attributes:

       **- tmp_logger (FileSystemLogger): Temporary logger for local logging**
           ..

       **before MLflow logging.**
           ..

   .. rubric:: Methods



   ========================================================  ==========
               **log_project(project: 'Project') -> None:**  Logs the project information to MLflow, including project name and parameters.  
   **log_metrics(metrics: dict[str, typing.Any]) -> None:**  Logs metrics to MLflow.  
            **log_artifact(artifact: dict, path) -> None:**  Logs artifacts to MLflow using the temporary logger.  
                      **log_params(params: dict) -> None:**  Logs parameters to MLflow.  
                        **log_model(model, path) -> None:**  Logs the model to MLflow using the temporary logger.  
   ========================================================  ==========

   ..
       !! processed by numpydoc !!
   .. py:method:: log_project(project: palma.base.project.Project) -> None


   .. py:method:: log_metrics(metrics: dict[str, Any]) -> None


   .. py:method:: log_artifact(artifact: dict, path) -> None


   .. py:method:: log_params(params: dict) -> None


   .. py:method:: log_model(model, path)



.. py:class:: _Logger(dummy)


   .. py:property:: logger
      :type: Logger


   .. py:method:: __set__(logger) -> None

      



      :Parameters:

          **logger: Logger**
              Define the logger to use.
              
              >>> from palma import logger, set_logger
              >>> from palma.components import FileSystemLogger
              >>> from palma.components import MLFlowLogger
              >>> set_logger(MLFlowLogger(uri="."))
              >>> set_logger(FileSystemLogger(uri="."))

          **Returns**
              ..

          **-------**
              None














      ..
          !! processed by numpydoc !!


