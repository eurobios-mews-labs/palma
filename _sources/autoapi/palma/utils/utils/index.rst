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


   .. py:method:: predict(*args, **kwargs) -> iter


   .. py:method:: predict_proba(*args, **kwargs) -> iter



.. py:function:: _clone(estimator)


.. py:function:: get_splitting_matrix(X: pandas.DataFrame, iter_cross_validation: iter, expand=False) -> pandas.DataFrame


.. py:function:: check_splitting_strategy(X: pandas.DataFrame, iter_cross_validation: iter)


.. py:function:: hash_dataframe(data: pandas.DataFrame, how='whole')


.. py:function:: get_hash(**kwargs) -> str

   
   Return a hash of parameters 
















   ..
       !! processed by numpydoc !!

.. py:function:: get_estimator_name(estimator) -> str


.. py:function:: check_started(message: str, need_build: bool = False) -> Callable

   
   check_built is a decorator used for methods that must be called on     built or unbuilt :class:`~autolm.project.Project`.
   If the :class:`~autolm.project.Project` is_built attribute has     not the correct value, an AttributeError is raised with the message passed     as argument.


   :Parameters:

       **message: str**
           Error message

       **need_build: bool**
           Expected value for :class:`~autolm.project.Project` is_built         attribute

   :Returns:

       Callable
           ..













   ..
       !! processed by numpydoc !!

.. py:function:: interpolate_roc(roc_curve_metric: dict[dict[tuple[dict[numpy.array]]]], mean_fpr=np.linspace(0, 1, 100))


.. py:function:: _get_processing_pipeline(estimators: list)


.. py:function:: _get_and_check_var_importance(estimator)


