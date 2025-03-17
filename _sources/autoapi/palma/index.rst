palma
=====

.. py:module:: palma


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/palma/base/index
   /autoapi/palma/components/index
   /autoapi/palma/datasets/index
   /autoapi/palma/preprocessing/index
   /autoapi/palma/utils/index


Attributes
----------

.. autoapisummary::

   palma.logger
   palma.set_logger


Classes
-------

.. autoapisummary::

   palma.ModelEvaluation
   palma.ModelSelector
   palma.Project


Package Contents
----------------

.. py:class:: ModelEvaluation(estimator)

   .. py:attribute:: __date


   .. py:attribute:: __model_id
      :value: ''



   .. py:attribute:: __estimator


   .. py:attribute:: __components


   .. py:attribute:: estimator_name


   .. py:attribute:: metrics


   .. py:method:: add(component, name=None)


   .. py:method:: fit(project: palma.base.project.Project)


   .. py:method:: __get_fit_estimators(X, y, indexes)


   .. py:method:: __compute_predictions(project, indexes, estimators)


   .. py:property:: id
      :type: str



   .. py:property:: components


   .. py:property:: unfit_estimator


.. py:class:: ModelSelector(engine: Union[str, palma.base.engine.BaseOptimizer], engine_parameters: Dict)

   
   Wrapper to optimizers selecting the best model for a Project.

   The optimization can be launched with the ``start`` method.
   Once the optimization is done, the best model can be accessed as the ``best_model_`` attribute.

   :Parameters:

       **- engine (str): Currently accepted values are "FlamlOptimizer" or**
           "AutoSklearnOptimizer" (the latter is deprecatted).

       **- engine_parameters (dict): parameters passed to the engine.**
           ..


   .. rubric:: Methods



   ==================================================  ==========
   **- start(project: Project): look for best model**    
   ==================================================  ==========












   ..
       !! processed by numpydoc !!

   .. py:attribute:: __date


   .. py:attribute:: __run_id
      :value: ''



   .. py:attribute:: __parameters


   .. py:method:: start(project: Project)


   .. py:property:: run_id
      :type: str



.. py:class:: Project(project_name: str, problem: str)

   
   Represents a machine learning project with various components
   and logging capabilities.


   :Parameters:

       **project_name (str): The name of the project.**
           ..

       **problem (str): The description of the machine learning problem.**
           Accepted values: "classification" or "regression".

   :Attributes:

       **base_index (List[int]): List of base indices for the project.**
           ..

       **components (dict): Dictionary containing project components.**
           ..

       **date (datetime): The date and time when the project was created.**
           ..

       **project_id (str): Unique identifier for the project.**
           ..

       **is_started (bool): Indicates whether the project has been started.**
           ..

       **problem (str): Description of the machine learning problem.**
           ..

       **validation_strategy (ValidationStrategy): The validation strategy used in the project.**
           ..

       **project_name (str): The name of the project.**
           ..

       **study_name (str): The randomly generated study name.**
           ..

       **X (pd.DataFrame): The feature data for the project.**
           ..

       **y (pd.Series): The target variable for the project.**
           ..

   .. rubric:: Methods



   ============================================================================================================  ==========
                                        **add(component: Component) -> None: Adds a component to the project.**    
   **start(X: pd.DataFrame, y: pd.Series, splitter, X_test=None, y_test=None, groups=None, **kwargs) -> None:**  Starts the project with the specified data and validation strategy.  
   ============================================================================================================  ==========












   ..
       !! processed by numpydoc !!

   .. py:attribute:: __project_name


   .. py:attribute:: __date


   .. py:attribute:: __study_name


   .. py:attribute:: __problem


   .. py:attribute:: __components


   .. py:attribute:: __is_started
      :value: False



   .. py:attribute:: __component_list
      :value: []



   .. py:method:: add(component: Component) -> None


   .. py:method:: start(X: pandas.DataFrame, y: pandas.Series, splitter, X_test=None, y_test=None, groups=None, **kwargs) -> None


   .. py:method:: __call_components(object_: Project) -> None


   .. py:property:: components
      :type: dict



   .. py:property:: date
      :type: datetime.datetime



   .. py:property:: project_id
      :type: str



   .. py:property:: is_started
      :type: bool



   .. py:property:: problem
      :type: str



   .. py:property:: validation_strategy
      :type: palma.base.splitting_strategy.ValidationStrategy



   .. py:property:: project_name
      :type: str



   .. py:property:: study_name
      :type: str



   .. py:property:: X
      :type: pandas.DataFrame



   .. py:property:: y
      :type: pandas.Series



.. py:data:: logger

.. py:data:: set_logger

