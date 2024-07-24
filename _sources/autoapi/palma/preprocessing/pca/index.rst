palma.preprocessing.pca
=======================

.. py:module:: palma.preprocessing.pca


Attributes
----------

.. autoapisummary::

   palma.preprocessing.pca.X


Classes
-------

.. autoapisummary::

   palma.preprocessing.pca.PCA


Module Contents
---------------

.. py:class:: PCA(data: pandas.DataFrame, prefix_name='pc')

   .. py:attribute:: data_train
      :type:  pandas.DataFrame


   .. py:attribute:: index


   .. py:attribute:: n


   .. py:attribute:: p


   .. py:attribute:: sc


   .. py:attribute:: scaled_data


   .. py:attribute:: pca


   .. py:attribute:: explained_variance


   .. py:attribute:: component_names


   .. py:attribute:: eigen_values


   .. py:attribute:: __nb_comp


   .. py:method:: set_nb_components(n=None, variance_threshold: float = None, **kwargs)


   .. py:property:: nb_component


   .. py:method:: transform(X: pandas.DataFrame) -> pandas.DataFrame


   .. py:method:: __get_corr(n_components=None) -> numpy.ndarray


   .. py:method:: get_correlation(n_components=None) -> pandas.DataFrame


   .. py:method:: get_variables_contributions(n_components=None) -> pandas.DataFrame


   .. py:method:: get_individual_contributions(n_components=None) -> pandas.DataFrame


   .. py:method:: plot_eigen_values() -> None


   .. py:method:: plot_cumulated_variance(color='tab:blue') -> None


   .. py:method:: plot_circle_corr() -> None


   .. py:method:: plot_correlation_matrix() -> None


   .. py:method:: plot_factorial_plan(X: pandas.DataFrame, x_axis='pc1', y_axis='pc2', c=None, cmap=None) -> None


   .. py:method:: plot_var_cp(X: pandas.DataFrame, n_col=3, figsize=(10, 10), x_axis='pc1', y_axis='pc2') -> None


   .. py:method:: plot_variance_bar(separator=0.5) -> None


.. py:data:: X

