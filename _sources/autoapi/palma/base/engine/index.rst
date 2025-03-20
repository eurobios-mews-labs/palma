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


   .. py:attribute:: __date


   .. py:attribute:: __run_id
      :value: ''



   .. py:attribute:: _problem
      :value: 'unknown'



   .. py:method:: optimize(X: pandas.DataFrame, y: pandas.Series, splitter: palma.base.splitting_strategy.ValidationStrategy = None) -> None
      :abstractmethod:



   .. py:property:: best_model_
      :type: None

      :abstractmethod:



   .. py:property:: transformer_
      :type: None

      :abstractmethod:



   .. py:property:: engine_parameters
      :type: Dict



   .. py:property:: allow_splitter


   .. py:method:: allowing_splitter(splitter)


   .. py:method:: start(project: Project)


   .. py:property:: run_id
      :type: str



   .. py:property:: problem


.. py:class:: FlamlOptimizer(engine_parameters: dict)

   Bases: :py:obj:`BaseOptimizer`


   .. py:method:: optimize(X: pandas.DataFrame, y: pandas.DataFrame, splitter: palma.base.splitting_strategy.ValidationStrategy = None) -> None


   .. py:property:: best_model_
      :type: sklearn.base.BaseEstimator



   .. py:property:: transformer_


   .. py:property:: allow_splitter


