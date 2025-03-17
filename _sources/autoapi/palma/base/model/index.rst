palma.base.model
================

.. py:module:: palma.base.model


Classes
-------

.. autoapisummary::

   palma.base.model.ModelEvaluation


Module Contents
---------------

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


