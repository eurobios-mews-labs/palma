Project
=======

The ``Project`` class is used to store how the training and validation will be carried out.
Two kind of problems are supported: ``regression`` and ``classification``.

An example in  regression_ and classification_

.. _regression: https://github.com/eurobios-mews-labs/palma/blob/main/examples/regression.ipynb
.. _classification: https://github.com/eurobios-mews-labs/palma/blob/main/examples/classification.ipynb

The idea of the project class is to fix before running any algorithm how
the training and validation will be carried out.

The setup is done in two steps: by initializing the class itself and then by calling the ``start`` method.
In the mean time, optional ``Component`` can be added with the ``add`` method.

.. warning::
    section in construction
