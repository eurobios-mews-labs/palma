:py:mod:`palma.components.performance`
======================================

.. py:module:: palma.components.performance


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.components.performance.Analyser
   palma.components.performance.ShapAnalysis
   palma.components.performance.ScoringAnalysis
   palma.components.performance.RegressionAnalysis




Attributes
~~~~~~~~~~

.. autoapisummary::

   palma.components.performance.colors
   palma.components.performance.colors_rainbow


.. py:data:: colors

   

.. py:data:: colors_rainbow

   

.. py:class:: Analyser(on)


   Bases: :py:obj:`palma.components.base.ModelComponent`

   
   Base Model Component class
















   ..
       !! processed by numpydoc !!
   .. py:attribute:: metrics

      

   .. py:attribute:: _metrics

      

   .. py:method:: __call__(project: Project, model: ModelEvaluation)


   .. py:method:: _add(project, model)


   .. py:method:: variable_importance()


   .. py:method:: compute_metrics(metric: dict)


   .. py:method:: _compute_metric(name: str, fun: Callable)

      
      Compute on specific metric and add it to 'metric' attribute
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_train_metrics() -> pandas.DataFrame


   .. py:method:: get_test_metrics() -> pandas.DataFrame


   .. py:method:: __get_metrics_helper(identifier) -> pandas.DataFrame


   .. py:method:: plot_variable_importance(mode='minmax', color='darkblue', cmap='flare')



.. py:class:: ShapAnalysis(on, n_shap, compute_interaction=False)


   Bases: :py:obj:`Analyser`

   
   Base Model Component class
















   ..
       !! processed by numpydoc !!
   .. py:method:: __call__(project: Project, model: ModelEvaluation)


   .. py:method:: __select_explainer()


   .. py:method:: _compute_shap_values(n, is_regression, explainer_method=shap.TreeExplainer, compute_interaction=False)


   .. py:method:: __change_features_name_to_string()


   .. py:method:: plot_shap_summary_plot()


   .. py:method:: plot_shap_decision_plot(**kwargs)


   .. py:method:: plot_shap_interaction(feature_x, feature_y)



.. py:class:: ScoringAnalysis(on)


   Bases: :py:obj:`Analyser`

   
   Base Model Component class
















   ..
       !! processed by numpydoc !!
   .. py:property:: threshold


   .. py:attribute:: mean_fpr

      

   .. py:method:: confusion_matrix(in_percentage=False)


   .. py:method:: __interpolate_roc(_)


   .. py:method:: plot_roc_curve(plot_method='mean', plot_train: bool = False, c=colors[0], cmap: str = 'inferno', cv_iter=None, label: str = '', mode: str = 'std', label_iter: iter = None, plot_base: bool = True, **kwargs)

      
      Plot the ROC curve.


      :Parameters:

          **plot_method** : str,
              Select the type of plot for ROC curve
              
              - "beam" (default) to plot all the curves using shades
              - "all" to plot each ROC curve
              - "mean" plot the mean ROC curve

          **plot_train: bool**
              If True the train ROC curves will be plot, default False.

          **c: str**
              Not used only with plot_method="all". Set the color of ROC curve

          **cmap: str**
              ..

          **cv_iter**
              ..

          **label**
              ..

          **mode**
              ..

          **label_iter**
              ..

          **plot_base: bool,**
              Plot basic ROC curve helper

          **kwargs:**
              Deprecated

      :Returns:

          
              ..













      ..
          !! processed by numpydoc !!

   .. py:method:: compute_threshold(method: str = 'total_population', value: float = 0.5, metric: Callable = None)

      
      Compute threshold using various heuristics
















      ..
          !! processed by numpydoc !!

   .. py:method:: plot_threshold(**plot_kwargs)



.. py:class:: RegressionAnalysis(on)


   Bases: :py:obj:`Analyser`

   
   Base Model Component class
















   ..
       !! processed by numpydoc !!
   .. py:method:: compute_predictions_errors(fun=None)


   .. py:method:: plot_prediction_versus_real(colormap=plot.get_cmap('rainbow'))


   .. py:method:: plot_errors_pairgrid(fun=None, number_percentiles=4, palette='rocket_r', features=None)



