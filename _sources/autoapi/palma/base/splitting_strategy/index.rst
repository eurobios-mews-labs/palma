:py:mod:`palma.base.splitting_strategy`
=======================================

.. py:module:: palma.base.splitting_strategy


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.base.splitting_strategy.ValidationStrategy



Functions
~~~~~~~~~

.. autoapisummary::

   palma.base.splitting_strategy._index_to_bool
   palma.base.splitting_strategy._bool_to_index



.. py:function:: _index_to_bool(array, length)


.. py:function:: _bool_to_index(array)


.. py:class:: ValidationStrategy(splitter: Union[sklearn.model_selection._split.BaseShuffleSplit, sklearn.model_selection._split.BaseCrossValidator, List[tuple], List[str]], **kwargs)


   
   Validation strategy for a machine learning project.


   :Parameters:

       **- splitter (Union[BaseShuffleSplit, BaseCrossValidator, List[tuple], List[str]]): The data splitting strategy.**
           ..












   :Attributes:

       **- test_index (np.ndarray): Index array for the test set.**
           ..

       **- train_index (np.ndarray): Index array for the training set.**
           ..

       **- indexes_val (list): List of indexes for validation sets.**
           ..

       **- indexes_train_test (list): List containing tuples of training and test indexes.**
           ..

       **- id: Unique identifier for the validation strategy.**
           ..

       **- splitter: The data splitting strategy.**
           ..

   .. rubric:: Methods



   ============================================================================================================================  ==========
   **- __call__(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None, groups=None, **kwargs):**  Applies the validation strategy to the provided data.  
   ============================================================================================================================  ==========

   ..
       !! processed by numpydoc !!
   .. py:property:: test_index
      :type: numpy.ndarray


   .. py:property:: train_index
      :type: numpy.ndarray


   .. py:property:: indexes_val
      :type: list


   .. py:property:: indexes_train_test
      :type: list


   .. py:property:: id


   .. py:property:: splitter


   .. py:property:: groups


   .. py:method:: __call__(X: pandas.DataFrame, y: pandas.Series, X_test: pandas.DataFrame = None, y_test: pandas.Series = None, groups=None, **kwargs)

      
      Apply the validation strategy to the provided data.
















      ..
          !! processed by numpydoc !!

   .. py:method:: __correct_nested(X)


   .. py:method:: __str__() -> str

      
      Return str(self).
















      ..
          !! processed by numpydoc !!


