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
   palma.components.logger.logger
   palma.components.logger.set_logger


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


   .. py:method:: log_artifact(**kwargs) -> None
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


   .. py:method:: log_artifact(obj, path: str) -> None



.. py:class:: FileSystemLogger(uri: str = tempfile.gettempdir(), **kwargs)


   Bases: :py:obj:`Logger`

   
   A logger for saving artifacts and metadata to the file system.


   :Parameters:

       **uri** : str, optional
           The root path or directory where artifacts and metadata will be saved.
           Defaults to the system temporary directory.

       **\*\*kwargs** : dict
           Additional keyword arguments to pass to the base logger.












   :Attributes:

       **path_project** : str
           The path to the project directory.

       **path_study** : str
           The path to the study directory within the project.

   .. rubric:: Methods



   ===================================================  ==========
             **log_project(project: Project) -> None**  Performs the first level of backup by creating folders and saving an instance of  :class:`~palma.Project`.  
     **log_metrics(metrics: dict, path: str) -> None**  Saves metrics in JSON format at the specified path.  
              **log_artifact(obj, path: str) -> None**  Saves an artifact at the specified path, handling different types of objects.  
   **log_params(parameters: dict, path: str) -> None**  Saves model parameters in JSON format at the specified path.  
   ===================================================  ==========

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

      
      Logs metrics to a JSON file.


      :Parameters:

          **metrics** : dict
              The metrics to be logged.

          **path** : str
              The relative path (from the study directory)
              where the metrics JSON file will be saved.














      ..
          !! processed by numpydoc !!

   .. py:method:: log_artifact(obj, path: str) -> None

      
      Logs an artifact, handling different types of objects.


      :Parameters:

          **obj** : any
              The artifact to be logged.

          **path** : str
              The relative path (from the study directory)
              where the artifact will be saved.














      ..
          !! processed by numpydoc !!

   .. py:method:: log_params(parameters: dict, path: str) -> None

      
      Logs model parameters to a JSON file.


      :Parameters:

          **parameters** : dict
              The model parameters to be logged.

          **path** : str
              The relative path (from the study directory) where the parameters
              JSON file will be saved.














      ..
          !! processed by numpydoc !!

   .. py:method:: __create_directories()

      
      Creates the study directory if it doesn't exist.

      If the study directory does not exist,
      it is created along with any necessary parent directories.















      ..
          !! processed by numpydoc !!


.. py:class:: MLFlowLogger(uri: str, artifact_location: str = '.mlruns')


   Bases: :py:obj:`Logger`

   
   MLFlowLogger class for logging experiments using MLflow.


   :Parameters:

       **uri** : str
           The URI for the MLflow tracking server.

       **artifact_location** : str
           The place to save artifact on file system logger





   :Raises:

       ImportError: If mlflow is not installed.
           ..







   :Attributes:

       **tmp_logger** : (FileSystemLogger)
           Temporary logger for local logging before MLflow logging.

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


   .. py:method:: log_metrics(metrics: dict[str, Any], path=None) -> None


   .. py:method:: log_artifact(artifact: dict, path) -> None


   .. py:method:: log_params(params: dict) -> None



.. py:class:: _Logger(dummy)


   .. py:property:: logger
      :type: Logger


   .. py:property:: uri


   .. py:method:: __set__(_logger) -> None

      



      :Parameters:

          **_logger: Logger**
              Define the logger to use.
              
              >>> from palma import logger, set_logger
              >>> from palma.components import FileSystemLogger
              >>> from palma.components import MLFlowLogger
              >>> set_logger(MLFlowLogger(uri="."))
              >>> set_logger(FileSystemLogger(uri="."))














      ..
          !! processed by numpydoc !!


.. py:data:: logger

   

.. py:data:: set_logger

   

