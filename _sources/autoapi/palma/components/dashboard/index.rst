:py:mod:`palma.components.dashboard`
====================================

.. py:module:: palma.components.dashboard


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.components.dashboard.ExplainerDashboard




Attributes
~~~~~~~~~~

.. autoapisummary::

   palma.components.dashboard.default_config_path


.. py:data:: default_config_path

   

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



