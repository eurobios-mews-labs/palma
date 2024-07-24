palma.base.engine
=================

.. py:module:: palma.base.engine


Classes
-------

.. autoapisummary::

   palma.base.engine.BaseOptimizer
   palma.base.engine.FlamlOptimizer


Module Contents
---------------

.. py:class:: BaseOptimizer(engine_parameters: dict)

   .. py:attribute:: __engine_parameters


   .. py:method:: optimize(X: pandas.DataFrame, y: pandas.Series, splitter: palma.base.splitting_strategy.ValidationStrategy = None) -> None
      :abstractmethod:



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


   .. py:method:: allowing_splitter(splitter)


.. py:class:: FlamlOptimizer(problem: str, engine_parameters: dict)

   Bases: :py:obj:`BaseOptimizer`


   .. py:method:: optimize(X: pandas.DataFrame, y: pandas.DataFrame, splitter: palma.base.splitting_strategy.ValidationStrategy = None) -> None


   .. py:property:: optimizer
      :type: flaml.AutoML



   .. py:property:: estimator_
      :type: sklearn.base.BaseEstimator



   .. py:property:: transformer_
      :type: flaml.data.DataTransformer



   .. py:property:: allow_splitter


