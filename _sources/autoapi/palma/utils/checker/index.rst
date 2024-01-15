:py:mod:`palma.utils.checker`
=============================

.. py:module:: palma.utils.checker


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.utils.checker.ProjectPlanChecker




Attributes
~~~~~~~~~~

.. autoapisummary::

   palma.utils.checker._CLASSIFICATION_METRICS
   palma.utils.checker._REGRESSION_METRICS


.. py:data:: _CLASSIFICATION_METRICS
   :value: ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score',...

   

.. py:data:: _REGRESSION_METRICS
   :value: ['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error',...

   

.. py:class:: ProjectPlanChecker


   Bases: :py:obj:`object`

   
   ProjectPlanChecker is an object that checks the project plan.

   At the :meth:`~palma.project.Project.build` moment, this object     run several checks in order to see if the project plan is well designed.

   Here is an overview of the checks performed by the object:
       - :meth:`~palma.utils.checker.ProjectPlanChecker._check_arrays`        : see whether X and y attribute are compliant with         sklearn standards.
       - :meth:`~palma.utils.checker.ProjectPlanChecker._check_project_problem`: see if the problem type is correctly         informed by the user.
       - :meth:`~palma.utils.checker.ProjectPlanChecker._check_problem_metrics`: see if the known metrics are consistent with         the project problem















   ..
       !! processed by numpydoc !!
   .. py:method:: _check_arrays(project: palma.Project) -> None


   .. py:method:: _check_project_problem(project: palma.Project) -> None


   .. py:method:: run_checks(project: palma.Project) -> None

      
      Perform some tests on the project plan

      Several checks are performed in order to check if the
      project plan is consistent:
          - checks the project problem 
          - checks the metrics provided by the user
          - checks the data provided by the user (scikit learn wrapper)

      :Parameters:

          **project** : :class:`~autolm.project.Project`
              an Project instance














      ..
          !! processed by numpydoc !!


