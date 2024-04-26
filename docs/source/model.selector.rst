Model selector
==============

A ``ModelSelector`` can be used to look for the best model.
This class is wrapper to either the optimizer from `FLAML <https://microsoft.github.io/FLAML/>`_ or the optimizer from `Auto-sklearn <https://automl.github.io/auto-sklearn/>`_.

The optimization can be launched with the ``start`` method.
Once the optimization is done, the best model can be accessed as the ``best_model_`` attribute.


.. warning::
    section in construction
