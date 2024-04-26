:py:mod:`palma.base.model_selection`
====================================

.. py:module:: palma.base.model_selection


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.base.model_selection.ModelSelector




.. py:class:: ModelSelector(engine: Union[str, palma.base.engine.BaseOptimizer], engine_parameters: Dict)


   
   Wrapper to optimizers selecting the best model for a Project.

   The optimization can be launched with the ``start`` method.
   Once the optimization is done, the best model can be accessed as the ``best_model_`` attribute.

   :Parameters:

       **- engine (str): Currently accepted values are "FlamlOptimizer" or**
           "AutoSklearnOptimizer" (the latter is deprecatted).

       **- engine_parameters (dict): parameters passed to the engine.**
           ..













   .. rubric:: Methods



   ==================================================  ==========
   **- start(project: Project): look for best model**    
   ==================================================  ==========

   ..
       !! processed by numpydoc !!
   .. py:property:: run_id
      :type: str


   .. py:method:: start(project: Project)



