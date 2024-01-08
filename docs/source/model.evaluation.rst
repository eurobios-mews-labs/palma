Model evaluation
================

Model fitting
-------------

This module fits any scikit-learn estimator on the train, validation and test
index specified on the project.

.. code-block:: python

   from palma import ModelEvaluation
   from sklearn.ensemble import RandomForestClassifier

   # Use your own estimator
   model = ModelEvaluation(estimator=RandomForestClassifier())
   model.fit(project)

   # Get the optimized estimator
   model = ModelEvaluation(estimator=ms.best_model_)
   model.fit(project)

The ModelEvaluation exposes several objects such as :code:`all_estimators_val_`
the list of all estimators fit on cross validation index.

Model analysis
--------------
To evaluate the performance of a model, you have to had analysis component first.

.. code-block:: python

    from palma import ModelEvaluation
    from palma.components import ScoringAnalysis
    from sklearn.ensemble import RandomForestClassifier

    # Use your own estimator
    model = ModelEvaluation(estimator=RandomForestClassifier())
    model.add(ScoringAnalysis())
    model.fit(project)
