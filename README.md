<img src=".static/logo.png" width="200"/>

### _Project for Automated Learning MAchine_ 

[![Maintenance](https://img.shields.io/badge/maintained%3F-yes-green.svg)](https://GitHub.com/eurobios-mews-labs/palma/graphs/commit-activity)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytest](https://github.com/eurobios-scb/palma/actions/workflows/pytest.yml/badge.svg?event=push)](https://docs.pytest.org)
[![PyPI version](https://badge.fury.io/py/palma.svg)](https://badge.fury.io/py/palma)
![code coverage](https://raw.githubusercontent.com/eurobios-mews-labs/palma/coverage-badge/coverage.svg?raw=true)

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

1. A vanilla approach described below (in basic usage section) and in the
   notebooks [classification](examples/classification.ipynb) and
   [regression](examples/regression.ipynb). In this approach, the users define
   a `Project`, which can then be passed to either a `ModelSelector` to find
   the best model for this project, or to a `ModelEvaluation` to study more in
   depth the behavior of a given model on this project.

2. A collection of [components](doc/components.md) that can be added to enrich
   analysis.

Install it with 
``` powershell
python -m pip install palma
```

## Documentation 

Access the [**full documentation here**](https://eurobios-mews-labs.github.io/palma/).

## Basic usage

1. Start your project

To start using the library, use the project class

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit
from palma import Project

X, y = make_classification(n_informative=2, n_features=100)
X, y = pd.DataFrame(X), pd.Series(y).astype(bool)

project = Project(problem="classification", project_name="default")

project.start(
    X, y,
    splitter=ShuffleSplit(n_splits=10, random_state=42),
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

This initialization is done in two steps to allow user to add optional
``Component``s to the project before its start.

2.  Run hyper-optimisation

The hyper-optimisation process will look for the best model in pool of models
that tend to perform well on various problem. For this specific task we make
use of [FLAML module](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML).
After hyper parametrisation, the metric to track can be computed

```python
from palma import ModelSelector

ms = ModelSelector(engine="FlamlOptimizer",
                   engine_parameters=dict(time_budget=30))
ms.start(project)
print(ms.best_model_)
```

3. Tailoring and analysing your estimator


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

## Contributing

You are very welcome to contribute to the project, by requesting features,
pointing out new tools that can be added as component, by identifying issues and creating new features. 
Development guidelines will be detailed in near future.

* Fork the repository
* Clone your forked repository ```git clone https://github.com/$USER/palma.git```
* Test using pytest ````pip install pytest; pytest tests/````
* Submit you work with a pull request.

## Authors

Eurobios Mews Labs

<img src=".static/logoEurobiosMewsLabs.png" width="150"/>
