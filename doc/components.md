# Components
In this library, adding components enable user to enrich his project by adding
(facultative) analysis. The idea is to incorporate quickly an analysis using 
the proposed framework. 

## Adding components
To add component, simply use the method `.add` of `project` or
 `model`

````python
import pandas as pd
import tempfile
from sklearn import model_selection
from sklearn.datasets import make_classification

from palma import Project
from palma import components

X, y = make_classification(n_informative=2, n_features=100)
X, y = pd.DataFrame(X), pd.Series(y).astype(bool)
splitter = model_selection.ShuffleSplit(n_splits=10, random_state=42)
project = Project(problem="classification", project_name="default")

project.add(components.FileSystemLogger(tempfile.gettempdir())) # Add an empty component
project.start(X, y, splitter=splitter)                          # Execute the __call__ of all added component
````



To add component to a model, simply use the api : 

````python
from palma import ModelEvaluation, components
from sklearn.ensemble import RandomForestClassifier
model = ModelEvaluation(estimator=RandomForestClassifier())
model.add(components.Component())   # Add an empty component
model.fit(project)                  # will execute the __call__ of all added components
````

## Access the components

To access the components (for instance those providing analysis method and plots),
do 
````python
component = model.components["Component"]
component
````

## Implemented components

**Associated to `Project`**
* `FileSystemLogger` : log data in the file system 
in a location provided in parameter
* `MLFlowLogger` : log data and project information using MLFlow
* `Profiler` : using Ydata profiling, create report. Requires logger
* `DeepChecks`: 

**Associated to `ModelEvaluation`**

- `ScoringAnalyser`: analysis tools for a classification problem
- `RegressionAnalyser`: analysis tools for a regression problem
- `ShapAnalyser`: local explanation using shap values

## Create your own component

Assume you have a function `fun(X, y, **parameters)`
that makes some analysis, then you can 
instantiate a component as such

````python
from palma.components.base import ProjectComponent  # import base class


def fun(X, y, **kwargs):
    return None


class MyComponent(ProjectComponent):
    def __init__(self, **parameters):
        self.parameters = parameters

    def __call__(self, project):
        print("Component properly called")
        return fun(project.X, project.y, **self.parameters)
````

and then use it in the following fashion

````python
project = Project(problem="classification", project_name="default")
project.add(MyComponent())   # Add an empty component
project.start(X, y, splitter=splitter)
````   

If you want to create a `ModelEvaluation`'s component, the signature of
the `__call__` is slightly different and should use both `projet` and `model`objects: 
`````python
    def __call__(self, project, model):
        print("Component properly called")
        return fun(project.X, project.y, **self.parameters)

`````
