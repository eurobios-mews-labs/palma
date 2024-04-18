:py:mod:`palma.components.data_checker`
=======================================

.. py:module:: palma.components.data_checker


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.components.data_checker.DeepCheck
   palma.components.data_checker.Leakage




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



