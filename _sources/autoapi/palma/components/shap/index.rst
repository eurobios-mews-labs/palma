palma.components.shap
=====================

.. py:module:: palma.components.shap


Classes
-------

.. autoapisummary::

   palma.components.shap.ShapAnalysis


Functions
---------

.. autoapisummary::

   palma.components.shap._select_explainer


Module Contents
---------------

.. py:class:: ShapAnalysis(on, n_shap, compute_interaction=False)

   Bases: :py:obj:`palma.components.performance.Analyser`


   
   Analyser class for performing analysis on a model.


   :Parameters:

       **on** : str
           The type of analysis to perform. Possible values are
           "indexes_train_test" or "indexes_val".














   ..
       !! processed by numpydoc !!

   .. py:attribute:: n_shap


   .. py:attribute:: compute_interaction
      :value: False



   .. py:method:: __call__(project: Project, model: ModelEvaluation)


   .. py:method:: _compute_shap_values(n, is_regression, compute_interaction=False)


   .. py:method:: __change_features_name_to_string()


   .. py:method:: plot_shap_summary_plot()


   .. py:method:: plot_shap_decision_plot(**kwargs)


   .. py:method:: plot_shap_interaction(feature_x, feature_y)


.. py:function:: _select_explainer(estimator)

