.. toctree::
   :maxdepth: 2
   :hidden:


Components
==========

In this library, adding components enables the user to enrich their project by adding (optional) analyses. The idea is to quickly incorporate an analysis using the proposed framework.


Adding Components
-----------------

To add a component, simply use the method `.add` of `project` or `model`:

.. code-block:: python

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

    project.add(components.FileSystemLogger(tempfile.gettempdir()))  # Add an empty component
    project.start(X, y, splitter=splitter)                           # Execute the __call__ of all added components

Accessing the Components
------------------------

To access the components (for instance, those providing analysis methods and plots), do:

.. code-block:: python

    component = model.components["Component"]
    component

.. image:: ../../.static/component.png
   :width: 500


Implemented Components
-----------------------

**Associated to `Project`**

- `FileSystemLogger`: log data in the file system in a location provided in parameters
- `MLFlowLogger`: log data and project information using MLFlow
- `Profiler`: using Ydata profiling, create a report. Requires logger
- `DeepChecks`

**Associated to `ModelEvaluation`**

- `ScoringAnalyser`: analysis tools for a classification problem
- `RegressionAnalyser`: analysis tools for a regression problem
- `ShapAnalyser`: local explanation using SHAP values

Creating Your Own Component
---------------------------

Assume you have a function `fun(X, y, **parameters)` that makes some analysis; then, you can instantiate a component as follows:

.. code-block:: python

    from palma.components.base import ProjectComponent  # import base class

    def fun(X, y, **kwargs):
        return None

    class MyComponent(ProjectComponent):
        def __init__(self, **parameters):
            self.parameters = parameters

        def __call__(self, project):
            print("Component properly called")
            return fun(project.X, project.y, **self.parameters)

And then use it in the following fashion:

.. code-block:: python

    project = Project(problem="classification", project_name="default")
    project.add(MyComponent())   # Add an empty component
    project.start(X, y, splitter=splitter)

If you want to create a `ModelEvaluation`'s component, the signature of the `__call__` is slightly different and should use both `project` and `model` objects:

.. code-block:: python

    def __call__(self, project, model):
        print("Component properly called")
        return fun(project.X, project.y, **self.parameters)
