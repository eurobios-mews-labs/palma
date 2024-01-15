:py:mod:`palma.datasets`
========================

.. py:module:: palma.datasets


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   palma.datasets.load_credit_approval



.. py:function:: load_credit_approval() -> Tuple[pandas.DataFrame, pandas.Series]

   
   Loads the Credit Approval dataset and prepares it for machine learning.

   The Credit Approval dataset is loaded using the uci_dataset library.
   The target variable 'A16' is transformed into binary labels, where '+'
   is encoded as True and '-' as False.
   Categorical features ('A13', 'A4', 'A6', 'A7', 'A5')
   are ordinal encoded using sklearn's OrdinalEncoder.
   Binary features ('A12', 'A9', 'A10', 'A1') are converted to boolean values.


   :Returns:

       Tuple[pd.DataFrame, pd.Series]
           A tuple containing the features (X) and target labels (y) for
            machine learning tasks.













   ..
       !! processed by numpydoc !!

