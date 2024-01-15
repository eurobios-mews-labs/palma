:py:mod:`palma.utils.utils`
===========================

.. py:module:: palma.utils.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.utils.utils.AverageEstimator



Functions
~~~~~~~~~

.. autoapisummary::

   palma.utils.utils._clone
   palma.utils.utils.get_splitting_matrix
   palma.utils.utils.check_splitting_strategy
   palma.utils.utils.hash_dataframe
   palma.utils.utils.get_hash
   palma.utils.utils.get_estimator_name
   palma.utils.utils.check_started
   palma.utils.utils.interpolate_roc
   palma.utils.utils._get_processing_pipeline
   palma.utils.utils._get_and_check_var_importance



.. py:class:: AverageEstimator(estimator_list: list)


   
   A simple ensemble estimator that computes the average prediction of a list of estimators.


   :Parameters:

       **estimator_list** : list
           A list of individual estimators to be averaged.

   :Returns:

       numpy.ndarray
           The averaged prediction or class probabilities.











   :Attributes:

       **estimator_list** : list
           The list of individual estimators.

       **n** : int
           The number of estimators in the list.

   .. rubric:: Methods



   ==================================  ==========
         **predict(*args, **kwargs)**  Compute the average prediction across all estimators.  
   **predict_proba(*args, **kwargs)**  Compute the average class probabilities across all estimators.  
   ==================================  ==========

   ..
       !! processed by numpydoc !!
   .. py:method:: predict(*args, **kwargs) -> iter


   .. py:method:: predict_proba(*args, **kwargs) -> iter



.. py:function:: _clone(estimator)

   
   Create and return a clone of the input estimator.


   :Parameters:

       **estimator** : object
           The estimator object to be cloned.

   :Returns:

       object
           A cloned copy of the input estimator.








   .. rubric:: Notes

   This function attempts to create a clone of the input estimator using the
   `clone` function. If the `clone` function is not available or raises a
   `TypeError`, it falls back to using `deepcopy`. If both methods fail, the
   original estimator is returned.


   .. rubric:: Examples

   >>> from sklearn.linear_model import LinearRegression
   >>> original_estimator = LinearRegression()
   >>> cloned_estimator = _clone(original_estimator)



   ..
       !! processed by numpydoc !!

.. py:function:: get_splitting_matrix(X: pandas.DataFrame, iter_cross_validation: iter, expand=False) -> pandas.DataFrame

   
   Generate a splitting matrix based on cross-validation iterations.


   :Parameters:

       **X** : pd.DataFrame
           The input dataframe.

       **iter_cross_validation** : Iterable
           An iterable containing cross-validation splits (train, test).

       **expand** : bool, optional
           If True, the output matrix will have columns for both train and test
           splits for each iteration. If False (default), the output matrix will
           have columns for each iteration with 1 for train and 2 for test.

   :Returns:

       pd.DataFrame
           A matrix indicating the train (1) and test (2) splits for each
           iteration. Rows represent data points, and columns represent iterations.










   .. rubric:: Examples

   >>> import pandas as pd
   >>> X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
   ...                   'feature2': ['A', 'B', 'C', 'D', 'E']})
   >>> iter_cv = [(range(3), range(3, 5)), (range(2), range(2, 5))]
   >>> get_splitting_matrix(X, iter_cv)



   ..
       !! processed by numpydoc !!

.. py:function:: check_splitting_strategy(X: pandas.DataFrame, iter_cross_validation: iter)


.. py:function:: hash_dataframe(data: pandas.DataFrame, how='whole')


.. py:function:: get_hash(**kwargs) -> str

   
   Return a hash of parameters 
















   ..
       !! processed by numpydoc !!

.. py:function:: get_estimator_name(estimator) -> str


.. py:function:: check_started(message: str, need_build: bool = False) -> Callable

   
   check_built is a decorator used for methods that must be called on     built or unbuilt :class:`~palma.Project`.
   If the :class:`~palma.Project` is_built attribute has     not the correct value, an AttributeError is raised with the message passed     as argument.


   :Parameters:

       **message: str**
           Error message

       **need_build: bool**
           Expected value for :class:`~palma.Project` is_built         attribute

   :Returns:

       Callable
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: interpolate_roc(roc_curve_metric: dict[dict[tuple[dict[numpy.array]]]], mean_fpr=np.linspace(0, 1, 100))


.. py:function:: _get_processing_pipeline(estimators: list)


.. py:function:: _get_and_check_var_importance(estimator)


