palma.automl.automl
===================

.. py:module:: palma.automl.automl


Attributes
----------

.. autoapisummary::

   palma.automl.automl.__default_project_component__
   palma.automl.automl.__default_model_component__
   palma.automl.automl.__default_regression_component__
   palma.automl.automl.__default_scoring_component__


Classes
-------

.. autoapisummary::

   palma.automl.automl.AutoMl


Module Contents
---------------

.. py:data:: __default_project_component__
   :value: []


.. py:data:: __default_model_component__

.. py:data:: __default_regression_component__

.. py:data:: __default_scoring_component__

.. py:class:: AutoMl(project_name: str, problem: str, X: pandas.DataFrame, y: pandas.Series, splitter, X_test=None, y_test=None, groups=None)

   
   AutoMl - Automated Machine Learning


   :Parameters:

       **project_name** : str
           Name of the machine learning project.

       **problem** : str
           Type of problem, either "classification" or "regression".

       **X** : pd.DataFrame
           Features of the training dataset.

       **y** : pd.Series
           Target variable of the training dataset.

       **splitter**
           Data splitter object for cross-validation.

       **X_test** : pd.DataFrame, optional
           Features of the test dataset, default is None.

       **y_test** : pd.Series, optional
           Target variable of the test dataset, default is None.

       **groups** : None, optional
           Grouping information for group-based cross-validation, default is None.

   :Attributes:

       **project** : Project
           Machine learning project object.

       **runner** : ModelSelector
           Model selection and training engine.

       **model** : ModelEvaluation
           Model evaluation and analysis object.

   .. rubric:: Methods



   ======================================  ==========
   **run(engine_name, engine_parameter)**  Run the automated machine learning process using the specified engine.  
   ======================================  ==========









   .. rubric:: Notes

   The `AutoMl` class is designed to automate the machine learning pipeline,
   including project setup, model selection, and evaluation.


   .. rubric:: Examples

   >>> automl = AutoMl(project_name='my-project',
   ...                 problem='classification',
   ...                 X=X,
   ...                 y=y,
   ...                 splitter=StratifiedKFold(n_splits=5))
   >>> automl.run(engine='FlamlEngine', engine_parameter={'time_budget': 20})

   ..
       !! processed by numpydoc !!

   .. py:attribute:: save_plt_backend


   .. py:attribute:: project


   .. py:method:: run(engine, engine_parameters)

      
      Run the automated machine learning process.


      :Parameters:

          **engine** : str
              Name of the engine to use.

          **engine_parameters**
              Parameters specific to the chosen machine learning engine.



      :Returns:

          self
              ..











      ..
          !! processed by numpydoc !!


