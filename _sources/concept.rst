The main concepts
=================

The Palma library aims at providing tools for an automatic machine learning approach. While many tools exist for individual components of AutoML, this library focuses on providing a structured framework rather than implementing a complete service.

In this library, a broad definition of AutoML is used, covering the optimization of hyperparameters, model historization, performance analysis, and any other element that can be replicated and must be included in the analysis results of the models.

Thanks to the use of components, this library is designed to be modular, allowing users to add their own analyses. It includes the following elements:

1. A vanilla approach described below (in the Basic Usage section) and in the notebooks [classification](examples/classification.ipynb) and [regression](examples/regression.ipynb).

2. A collection of :doc:`components` that can be added to enrich the analysis.