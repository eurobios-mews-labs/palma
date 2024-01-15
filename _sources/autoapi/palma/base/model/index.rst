:py:mod:`palma.base.model`
==========================

.. py:module:: palma.base.model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.base.model.ModelEvaluation




.. py:class:: ModelEvaluation(estimator)


   .. py:property:: id
      :type: str


   .. py:property:: components


   .. py:method:: add(component, name=None)


   .. py:method:: fit(project: palma.base.project.Project)


   .. py:method:: __get_fit_estimators(X, y, indexes)


   .. py:method:: __compute_predictions(project, indexes)



