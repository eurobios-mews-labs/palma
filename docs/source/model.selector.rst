Model selector
==============

A ``ModelSelector`` can be used to look for the best model.
This class is wrapper to either the optimizer from `FLAML <https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/>`_ or the optimizer from `Auto-sklearn <https://automl.github.io/auto-sklearn/>`_ (the latter is deprecated).

The optimization can be launched with the ``start`` method.
Once the optimization is done, the best model can be accessed as the ``best_model_`` attribute.


.. warning::
    section in construction
