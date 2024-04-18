:py:mod:`palma.components`
==========================

.. py:module:: palma.components


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base/index.rst
   checker/index.rst
   dashboard/index.rst
   data_checker/index.rst
   data_profiler/index.rst
   logger/index.rst
   performance/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   palma.components.Component
   palma.components.FileSystemLogger
   palma.components.MLFlowLogger
   palma.components.ProfilerYData
   palma.components.ExplainerDashboard
   palma.components.RegressionAnalysis
   palma.components.ScoringAnalysis
   palma.components.ShapAnalysis
   palma.components.DeepCheck
   palma.components.Leakage




.. py:class:: Component


   Bases: :py:obj:`object`

   .. py:method:: __str__()

      
      Return str(self).
















      ..
          !! processed by numpydoc !!


.. py:class:: FileSystemLogger(uri: str = tempfile.gettempdir(), **kwargs)


   Bases: :py:obj:`Logger`

   
   A logger for saving artifacts and metadata to the file system.


   :Parameters:

       **uri** : str, optional
           The root path or directory where artifacts and metadata will be saved.
           Defaults to the system temporary directory.

       **\*\*kwargs** : dict
           Additional keyword arguments to pass to the base logger.












   :Attributes:

       **path_project** : str
           The path to the project directory.

       **path_study** : str
           The path to the study directory within the project.

   .. rubric:: Methods



   ===================================================  ==========
             **log_project(project: Project) -> None**  Performs the first level of backup by creating folders and saving an instance of  :class:`~palma.Project`.  
     **log_metrics(metrics: dict, path: str) -> None**  Saves metrics in JSON format at the specified path.  
              **log_artifact(obj, path: str) -> None**  Saves an artifact at the specified path, handling different types of objects.  
   **log_params(parameters: dict, path: str) -> None**  Saves model parameters in JSON format at the specified path.  
   ===================================================  ==========

   ..
       !! processed by numpydoc !!
   .. py:method:: log_project(project: palma.base.project.Project) -> None

      
      log_project performs the first level of backup as described
      in the object description. 

      This method creates the needed folders and saves an instance of         :class:`~palma.Project`.

      :Parameters:

          **project: :class:`~palma.Project`**
              an instance of Project














      ..
          !! processed by numpydoc !!

   .. py:method:: log_metrics(metrics: dict, path: str) -> None

      
      Logs metrics to a JSON file.


      :Parameters:

          **metrics** : dict
              The metrics to be logged.

          **path** : str
              The relative path (from the study directory)
              where the metrics JSON file will be saved.














      ..
          !! processed by numpydoc !!

   .. py:method:: log_artifact(obj, path: str) -> None

      
      Logs an artifact, handling different types of objects.


      :Parameters:

          **obj** : any
              The artifact to be logged.

          **path** : str
              The relative path (from the study directory)
              where the artifact will be saved.














      ..
          !! processed by numpydoc !!

   .. py:method:: log_params(parameters: dict, path: str) -> None

      
      Logs model parameters to a JSON file.


      :Parameters:

          **parameters** : dict
              The model parameters to be logged.

          **path** : str
              The relative path (from the study directory) where the parameters
              JSON file will be saved.














      ..
          !! processed by numpydoc !!

   .. py:method:: __create_directories()

      
      Creates the study directory if it doesn't exist.

      If the study directory does not exist,
      it is created along with any necessary parent directories.















      ..
          !! processed by numpydoc !!


.. py:class:: MLFlowLogger(uri: str, artifact_location: str = '.mlruns')


   Bases: :py:obj:`Logger`

   
   MLFlowLogger class for logging experiments using MLflow.


   :Parameters:

       **uri** : str
           The URI for the MLflow tracking server.

       **artifact_location** : str
           The place to save artifact on file system logger





   :Raises:

       ImportError: If mlflow is not installed.
           ..







   :Attributes:

       **tmp_logger** : (FileSystemLogger)
           Temporary logger for local logging before MLflow logging.

   .. rubric:: Methods



   ========================================================  ==========
               **log_project(project: 'Project') -> None:**  Logs the project information to MLflow, including project name and parameters.  
   **log_metrics(metrics: dict[str, typing.Any]) -> None:**  Logs metrics to MLflow.  
            **log_artifact(artifact: dict, path) -> None:**  Logs artifacts to MLflow using the temporary logger.  
                      **log_params(params: dict) -> None:**  Logs parameters to MLflow.  
                        **log_model(model, path) -> None:**  Logs the model to MLflow using the temporary logger.  
   ========================================================  ==========

   ..
       !! processed by numpydoc !!
   .. py:method:: log_project(project: palma.base.project.Project) -> None


   .. py:method:: log_metrics(metrics: dict[str, Any], path=None) -> None


   .. py:method:: log_artifact(artifact: dict, path) -> None


   .. py:method:: log_params(params: dict) -> None



.. py:class:: ProfilerYData(**config)


   Bases: :py:obj:`palma.components.base.ProjectComponent`

   
   Base Project Component class

   This object ensures that all subclasses Project component implements a















   ..
       !! processed by numpydoc !!
   .. py:method:: __call__(project: Project)



.. py:class:: ExplainerDashboard(dashboard_config: Union[str, Dict] = default_config_path, n_sample: int = None)


   Bases: :py:obj:`palma.components.base.Component`

   .. py:method:: __call__(project: Project, model: Model) -> explainerdashboard.ExplainerDashboard

      
      This function returns dashboard instance. This dashboard is to be run
      using its `run` method.


      :Parameters:

          **project: Project**
              Instance of project used to compute explainer.

          **model: Run**
              Current run to use in explainer.











      .. rubric:: Examples

      >>> db = ExpDash(dashboard_config="path_to_my_config")
      >>> explainer_dashboard = db(project, model)
      >>> explainer_dashboard.run(
      >>>    port="8050", host="0.0.0.0", use_waitress=False)



      ..
          !! processed by numpydoc !!

   .. py:method:: update_config(dict_value: Dict[str, Dict])

      
      Update specific parameters from the actual configuration.


      :Parameters:

          **dict_value: dict**
              explainer_parameters: dict
                  Parameters to be used in see `explainerdashboard.RegressionExplainer`
                  or `explainerdashboard.ClassifierExplainer`.
              dashboard_parameters: dict
                  Parameters use to compose dashboard tab, items or themes
                  for `explainerdashboard.ExplainerDashboard`.
                  Tabs and component of the dashboard can be hidden, see
                  `customize dashboard section <https://explainerdashboard.readthedocs.io/en/latest/custom.html>`_
                  for more detail.














      ..
          !! processed by numpydoc !!

   .. py:method:: _prepare_dataset() -> None

      
      This function performs the following processing steps :
          - Ensure that column name is str (bug encountered in dashboard)
          - Get code from categories just in case of category data types
          - Sample the data if specified by user
















      ..
          !! processed by numpydoc !!

   .. py:method:: _get_explainer(project: Project, model: Model) -> explainerdashboard.explainers.BaseExplainer


   .. py:method:: _get_dashboard(explainer: explainerdashboard.explainers.BaseExplainer) -> ExplainerDashboard



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



.. py:class:: DeepCheck(name: str = 'Data Checker', dataset_parameters: dict = None, dataset_checks: Union[List[deepchecks.core.BaseCheck], deepchecks.core.BaseSuite] = data_integrity(), train_test_datasets_checks: Union[List[deepchecks.core.BaseCheck], deepchecks.core.BaseSuite] = Suite('Checks train test', train_test_validation()), raise_on_fail=True)


   Bases: :py:obj:`palma.components.base.ProjectComponent`

   
   This object is a wrapper of the Deepchecks library and allows to audit the
   data through various checks such as data drift, duplicate values, ...


   :Parameters:

       **dataset_parameters** : dict, optional
           Parameters and their values that will be used to generate
           :class:`deepchecks.Dataset` instances (required to run the checks on)

       **dataset_checks: Union[List[BaseCheck], BaseSuite], optional**
           List of checks or suite of checks that will be run on the whole dataset
           By default: use the default suite single_dataset_integrity to detect
           the integrity issues

       **train_test_datasets_checks: Union[List[BaseCheck], BaseSuite], optional**
           List of checks or suite of checks to detect issues related to the
           train-test split, such as feature drift, detecting data leakage...
           By default, use the default suites train_test_validation and
           train_test_leakage

       **raise_on_fail: bool, optional**
           Raises error if one test fails














   ..
       !! processed by numpydoc !!
   .. py:method:: __call__(project: palma.base.project.Project) -> None

      
      Run suite of checks on the project data.


      :Parameters:

          **project: :class:`~palma.Project`**
              ..














      ..
          !! processed by numpydoc !!

   .. py:method:: __generate_datasets(project: palma.base.project.Project, **kwargs) -> None

      
      Generate :class:`deepchecks.Dataset`


      :Parameters:

          **project: project**
              :class:`~palma.Project`














      ..
          !! processed by numpydoc !!

   .. py:method:: __generate_suite(checks: Union[List[deepchecks.core.BaseCheck], deepchecks.core.BaseSuite], name: str) -> deepchecks.tabular.Suite
      :staticmethod:

      
      Generate a Suite of checks from a list of checks or a suite of checks


      :Parameters:

          **checks: Union[List[BaseCheck], BaseSuite], optional**
              List of checks or suite of checks

          **name: str**
              Name for the suite to returned

      :Returns:

          suite: :class:`deepchecks.Suite`
              instance of :class:`deepchecks.Suite`













      ..
          !! processed by numpydoc !!


.. py:class:: Leakage


   Bases: :py:obj:`palma.components.base.ProjectComponent`

   
   Class for detecting data leakage in a classification project.

   This class implements component that checks for data leakage in a given
   project. It uses the FLAML optimizer for model selection and performs
   a scoring analysis to check for the presence of data leakage based on
   the AUC metric.















   ..
       !! processed by numpydoc !!
   .. py:property:: metrics


   .. py:method:: __call__(project: palma.base.project.Project) -> None


   .. py:method:: cross_validation_leakage(project)



