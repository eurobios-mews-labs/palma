:py:mod:`palma.base.engine`
===========================

.. py:module:: palma.base.engine


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.base.engine.BaseOptimizer
   palma.base.engine.AutoSklearnOptimizer
   palma.base.engine.FlamlOptimizer




.. py:class:: BaseOptimizer(engine_parameters: dict)


   Bases: :py:obj:`object`

   .. py:property:: optimizer
      :type: None
      :abstractmethod:


   .. py:property:: estimator_
      :type: None
      :abstractmethod:


   .. py:property:: transformer_
      :type: None
      :abstractmethod:


   .. py:property:: engine_parameters
      :type: Dict


   .. py:property:: allow_splitter


   .. py:method:: optimize(X: pandas.DataFrame, y: pandas.Series, splitter=None) -> None
      :abstractmethod:


   .. py:method:: allowing_splitter(splitter)



.. py:class:: AutoSklearnOptimizer(problem: str, engine_parameters: dict)


   Bases: :py:obj:`BaseOptimizer`

   .. py:property:: optimizer
      :type: Union[autosklearn.classification.AutoSklearnClassifier, autosklearn.regression.AutoSklearnRegressor]


   .. py:property:: estimator_
      :type: Union[autosklearn.classification.AutoSklearnClassifier, autosklearn.regression.AutoSklearnRegressor]


   .. py:property:: transformer_


   .. py:method:: optimize(X: pandas.DataFrame, y: pandas.Series, splitter=None) -> None



.. py:class:: FlamlOptimizer(problem: str, engine_parameters: dict)


   Bases: :py:obj:`BaseOptimizer`

   .. py:property:: optimizer
      :type: flaml.AutoML


   .. py:property:: estimator_
      :type: sklearn.base.BaseEstimator


   .. py:property:: transformer_
      :type: flaml.data.DataTransformer


   .. py:property:: allow_splitter


   .. py:method:: optimize(X: pandas.DataFrame, y: pandas.DataFrame, splitter=None) -> None



