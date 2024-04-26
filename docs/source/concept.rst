The main concepts
=================

The Palma library aims at providing tools for an automatic machine learning approach. While many tools exist for individual components of AutoML, this library focuses on providing a structured framework rather than implementing a complete service.

In this library, a broad definition of AutoML is used, covering the optimization of hyperparameters, model historization, performance analysis, and any other element that can be replicated and must be included in the analysis results of the models.

Thanks to the use of components, this library is designed to be modular, allowing users to add their own analyses. It includes the following elements:

1. A vanilla approach in the Basic Usage section and in the notebooks [classification](https://github.com/eurobios-mews-labs/palma/blob/main/examples/classification.ipynb) and [regression](https://github.com/eurobios-mews-labs/palma/blob/main/examples/regression.ipynb). In this approach, the users define a :doc:`Project`, which can then be passed to either a :doc:`Model selector` to find the best model for this project, or to a :doc:`Model evaluation` to study more in depth the behavior of a given model on this project.

2. A collection of :doc:`components` that can be added to enrich the analysis. They can be added to a ``Project`` or to a ``ModelEvaluation``.
