# PALMA 
### _Project for Automated Learning MAchine_ 

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/eurobios-mews-labs/palma/graphs/commit-activity)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytest](https://github.com/eurobios-scb/palma/actions/workflows/pytest.yml/badge.svg?event=push)](https://docs.pytest.org)

This library aims at providing tools for an automatic machine learning approach.
As many tools already exist to establish one or the other component of an AutoML
approach, the idea of this library is to provide a structure rather than to
implement a complete service.
In this library, a broad definition of AutoML is used : it covers the
optimization of hyperparameters, the historization of models, the analysis
of performances etc. In short, any element that can be replicated and that must,
in most cases, be included in the analysis results of the models.
Also, thanks to the use of components, this
library is designed to be modular and allows the user to add his own
analyses.    
It therefore contains the following elements

1. A vanilla approach described below (in basic usage section) and in the notebooks
[classification](examples/classification.ipynb) and [regression](examples/regression.ipynb)

2. A collection of [components](doc/components.md) that can be added to enrich
   analysis

## Install notice

``` powershell
python -m pip install git+https://github.com/eurobios-mews-labs/palma.git
```

## Basic usage

### Start your project

To start using the library, use the project class

```python
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
```

The instantiation defines the type of problem and the `start` method will set
what is needed to carry out ML project :

- A testing strategy (argument `splitter`). That will define train and test
  instances.
  Note that we use cross validator from sklearn to do that. In the
  optimisation of hyper-parameters, a train test split will be operated, in this
  case, the first split will be used.
  This implies for instance that if you want 80/20 splitting method that shuffle
  the dataset, you should use

```python
splitter = model_selection.ShuffleSplit(n_splits=5, random_state=42)
```

- Training data `X` and target `y`

### Run hyper-optimisation

The hyper-optimisation process will look for the best model in pool of models
that tend to perform well on various problem.
For this specific task we make use of FLAML module. After hyper parametrisation,
the metric to track can be computed

```python
from palma import ModelSelector

ms = ModelSelector(engine="FlamlOptimizer",
                   engine_parameters=dict(time_budget=30))
ms.start(project)
print(ms.best_model_)
```

### Tailoring and analysing your estimator


```python
from palma import ModelEvaluation
from sklearn.ensemble import RandomForestClassifier

# Use your own
model = ModelEvaluation(estimator=RandomForestClassifier())
model.fit(project)

# Get the optimized estimator
model = ModelEvaluation(estimator=ms.best_model_)
model.fit(project)
```

### Manage components

You can add component to enrich the project.
See [here](doc/components.md) for a detailed documentation.

## Authors

Eurobios Mews Labs

<img src="doc/logoEurobiosMewsLabs.png" width="200"/>
