:py:mod:`palma.utils.plotting`
==============================

.. py:module:: palma.utils.plotting


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   palma.utils.plotting.plot_correlation
   palma.utils.plotting.plot_splitting_strategy
   palma.utils.plotting.plot_variable_importance
   palma.utils.plotting.roc_plot_bundle
   palma.utils.plotting.roc_plot_base



.. py:function:: plot_correlation(df: pandas.DataFrame, cmap: str = 'RdBu_r', method: str = 'spearman', linewidths=1, fmt='0.2f', vmin=-1, vmax=1)


.. py:function:: plot_splitting_strategy(X: pandas.DataFrame, y: pandas.Series, iter_cross_validation: iter, cmap, sort_by=None, modulus=1)


.. py:function:: plot_variable_importance(variable_importance: pandas.DataFrame, mode='minmax', color='darkblue', cmap='flare', alpha=0.2)


.. py:function:: roc_plot_bundle(list_fpr, list_tpr, mean_fpr=np.linspace(0, 1, 100), plot_all=False, plot_beam=True, cmap='inferno', plot_mean=True, c='b', label_iter=None, mode='std', label='', **args)


.. py:function:: roc_plot_base()


