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

   palma.components.performance.__fpr_sampling__


.. py:data:: __fpr_sampling__

   

.. py:class:: Analyser(on)


   Bases: :py:obj:`palma.components.base.ModelComponent`

   
   Analyser class for performing analysis on a model.


   :Parameters:

       **on** : str
           The type of analysis to perform. Possible values are
           "indexes_train_test" or "indexes_val".














   ..
       !! processed by numpydoc !!
   .. py:property:: metrics


   .. py:method:: __call__(project: Project, model: ModelEvaluation)


   .. py:method:: _add(project, model)


   .. py:method:: variable_importance()

      
      Compute the feature importance for each estimator.



      :Returns:

          **feature_importance** : pandas.DataFrame
              DataFrame containing the feature importance values for each estimator.













      ..
          !! processed by numpydoc !!

   .. py:method:: compute_metrics(metric: dict)

      
      Compute the specified metrics for each estimator.


      :Parameters:

          **metric** : dict
              Dictionary containing the metric name as key and the metric function as value.














      ..
          !! processed by numpydoc !!

   .. py:method:: _compute_metric(name: str, fun: Callable)

      
      Compute a specific metric and add it to the metrics attribute.


      :Parameters:

          **name** : str
              The name of the metric.

          **fun** : callable
              The function to compute the metric.














      ..
          !! processed by numpydoc !!

   .. py:method:: get_train_metrics() -> pandas.DataFrame

      
      Get the computed metrics for the training set.



      :Returns:

          pd.DataFrame
              DataFrame containing the computed metrics for the training set.













      ..
          !! processed by numpydoc !!

   .. py:method:: get_test_metrics() -> pandas.DataFrame

      
      Get the computed metrics for the test set.



      :Returns:

          pd.DataFrame
              DataFrame containing the computed metrics for the test set.













      ..
          !! processed by numpydoc !!

   .. py:method:: __get_metrics_helper(identifier) -> pandas.DataFrame


   .. py:method:: plot_variable_importance(mode='minmax', color='darkblue', cmap='flare', **kwargs)

      
      Plot the variable importance.


      :Parameters:

          **mode** : str, optional
              The mode for plotting the variable importance, by default "minmax".

          **color** : str, optional
              The color for the plot, by default "darkblue".

          **cmap** : str, optional
              The colormap for the plot, by default "flare".














      ..
          !! processed by numpydoc !!


.. py:class:: ShapAnalysis(on, n_shap, compute_interaction=False)


   Bases: :py:obj:`Analyser`

   
   Analyser class for performing analysis on a model.


   :Parameters:

       **on** : str
           The type of analysis to perform. Possible values are
           "indexes_train_test" or "indexes_val".














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

   
   The ScoringAnalyser class provides methods for analyzing the performance of
   a machine learning model.
















   ..
       !! processed by numpydoc !!
   .. py:property:: threshold


   .. py:method:: confusion_matrix(in_percentage=False)

      
      Compute the confusion matrix.


      :Parameters:

          **in_percentage** : bool, optional
              Whether to return the confusion matrix in percentage, by default False

      :Returns:

          pandas.DataFrame
              The confusion matrix













      ..
          !! processed by numpydoc !!

   .. py:method:: __interpolate_roc(_)


   .. py:method:: plot_roc_curve(plot_method='mean', plot_train: bool = False, c='C0', cmap: str = 'inferno', label: str = '', mode: str = 'std', label_iter: iter = None, plot_base: bool = True, **kwargs)

      
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


      :Parameters:

          **method** : str, optional
              The method to compute the threshold, by default "total_population"
              
              - total population : compute threshold so that the percentage of
              positive prediction is equal to `value`
              - fpr : compute threshold so that the false positive rate
              is equal to `value`
              - optimize_metric : compute threshold so that the metric is optimized
              `value` parameter is ignored, `metric` parameter must be provided

          **value** : float, optional
              The value to use for the threshold computation, by default 0.5

          **metric** : typing.Callable, optional
              The metric function to use for the threshold computation, by default None

      :Returns:

          float
              The computed threshold













      ..
          !! processed by numpydoc !!

   .. py:method:: plot_threshold(**plot_kwargs)

      
      Plot the threshold on fpr/tpr axes


      :Parameters:

          **plot_kwargs** : dict, optional
              Additional keyword arguments to pass to the scatter plot function

      :Returns:

          matplotlib.pyplot
              The threshold plot













      ..
          !! processed by numpydoc !!


.. py:class:: RegressionAnalysis(on)


   Bases: :py:obj:`Analyser`

   
   Analyser class for performing analysis on a regression model.


   :Parameters:

       **on** : str
           The type of analysis to perform. Possible values are
           "indexes_train_test" or "indexes_val".












   :Attributes:

       **_hidden_metrics** : dict
           Dictionary to store additional metrics that are not displayed.

   .. rubric:: Methods



   ===========================================================================  ==========
                                                     **variable_importance()**  Compute the feature importance for each estimator.  
                                             **compute_metrics(metric: dict)**  Compute the specified metrics for each estimator.  
                                       **get_train_metrics() -> pd.DataFrame**  Get the computed metrics for the training set.  
                                        **get_test_metrics() -> pd.DataFrame**  Get the computed metrics for the test set.  
   **plot_variable_importance(mode="minmax", color="darkblue", cmap="flare")**  Plot the variable importance.  
                                               **plot_prediction_versus_real**  Plot prediction versus real values  
                                                      **plot_errors_pairgrid**  Plot pair grid errors  
   ===========================================================================  ==========

   ..
       !! processed by numpydoc !!
   .. py:method:: compute_predictions_errors(fun=None)


   .. py:method:: plot_prediction_versus_real(colormap=plot.get_cmap('rainbow'))


   .. py:method:: plot_errors_pairgrid(fun=None, number_percentiles=4, palette='rocket_r', features=None)



