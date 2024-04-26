:py:mod:`palma.base.project`
============================

.. py:module:: palma.base.project


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.base.project.Project




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


   .. py:method:: add(component: Component) -> None


   .. py:method:: start(X: pandas.DataFrame, y: pandas.Series, splitter, X_test=None, y_test=None, groups=None, **kwargs) -> None


   .. py:method:: __call_components(object_: Project) -> None



