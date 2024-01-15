:py:mod:`palma.components.base`
===============================

.. py:module:: palma.components.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.components.base.Component
   palma.components.base.ProjectComponent
   palma.components.base.ModelComponent




.. py:class:: Component


   Bases: :py:obj:`object`

   .. py:method:: __str__()

      
      Return str(self).
















      ..
          !! processed by numpydoc !!


.. py:class:: ProjectComponent


   Bases: :py:obj:`Component`

   
   Base Project Component class

   This object ensures that all subclasses Project component implements a















   ..
       !! processed by numpydoc !!
   .. py:method:: __call__(project: palma.base.project.Project) -> None
      :abstractmethod:



.. py:class:: ModelComponent


   Bases: :py:obj:`Component`

   
   Base Model Component class
















   ..
       !! processed by numpydoc !!
   .. py:method:: __call__(project: palma.base.project.Project, model)
      :abstractmethod:



