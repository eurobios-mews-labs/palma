.. palma documentation master file, created by
   sphinx-quickstart on Thu Jan  4 17:25:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: ../../.static/logo.png
   :width: 200

\

.. image:: https://img.shields.io/badge/maintained%3F-yes-green.svg
   :target: https://GitHub.com/eurobios-mews-labs/palma/graphs/commit-activity

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit

.. image:: https://github.com/eurobios-scb/palma/actions/workflows/pytest.yml/badge.svg?event=push
   :target: https://docs.pytest.org

.. image:: https://badge.fury.io/py/palma.svg
   :target: https://badge.fury.io/py/palma

* The Palma library aims to provide simple tools to accelerate the development of your machine learning project

Installation
------------

To install the Palma library, use the following command:

.. code-block:: powershell

   python -m pip install palma

Basic Usage
-----------

Start your project by using the project class:

.. code-block:: python

   import pandas as pd
   from sklearn import model_selection
   from sklearn.datasets import make_classification
   from palma import Project

   X, y = make_classification(n_informative=2, n_features=100)
   X, y = pd.DataFrame(X), pd.Series(y).astype(bool)
   project = Project(problem="classification", project_name="default")
   project.start(
       X, y,
       splitter=model_selection.ShuffleSplit(n_splits=10, random_state=42),
   )

The instantiation defines the type of problem, and the `start` method will set up what is needed to carry out an ML project, including a testing strategy (argument `splitter`), training data `X`, and target `y`.

Run Hyper-optimization
----------------------

.. image:: ../../.static/hyperopt.png
   :width: 800


The hyper-optimization process will look for the best model in a pool of models that tend to perform well on various problems. For this specific task, the FLAML module is used. After hyperparameterization, the metric to track can be computed:

.. code-block:: python

   from palma import ModelSelector

   ms = ModelSelector(engine="FlamlOptimizer",
                      engine_parameters=dict(time_budget=30))
   ms.start(project)
   print(ms.best_model_)

Tailoring and Analyzing Your Estimator
--------------------------------------

.. code-block:: python

   from palma import ModelEvaluation
   from sklearn.ensemble import RandomForestClassifier

   # Use your own estimator
   model = ModelEvaluation(estimator=RandomForestClassifier())
   model.fit(project)

   # Get the optimized estimator
   model = ModelEvaluation(estimator=ms.best_model_)
   model.fit(project)

Manage Components
-----------------

You can add components to enrich the project. See :doc:`components` for detailed documentation.

Authors
-------

Eurobios Mews Labs

.. image:: ../../.static/logoEurobiosMewsLabs.png
   :width: 150


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Concepts

   concept


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Basic components

   project
   model.selector
   model.evaluation


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced Usage

   components

