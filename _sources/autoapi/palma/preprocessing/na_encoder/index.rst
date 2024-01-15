:py:mod:`palma.preprocessing.na_encoder`
========================================

.. py:module:: palma.preprocessing.na_encoder


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   palma.preprocessing.na_encoder.NA_encoder




.. py:class:: NA_encoder(numerical_strategy='mean', categorical_strategy='<NULL>')


   
   Encodes missing values for both numerical and categorical features.

   Several strategies are possible in each case.

   :Parameters:

       **numerical_strategy** : str or float or int. default = "mean"
           The strategy to encode NA for numerical features.
           Available strategies = "mean", "median",
           "most_frequent" or a float/int value

       **categorical_strategy** : str, default = '<NULL>'
           The strategy to encode NA for categorical features.
           Available strategies = a string or "most_frequent"














   ..
       !! processed by numpydoc !!
   .. py:method:: get_params(deep=True)

      
      Get parameters of a NA_encoder object.
















      ..
          !! processed by numpydoc !!

   .. py:method:: set_params(**params)

      
      Set parameters for a NA_encoder object.

      Set numerical strategy and categorical strategy.

      :Parameters:

          **numerical_strategy** : str or float or int. default = "mean"
              The strategy to encode NA for numerical features.

          **categorical_strategy** : str, default = '<NULL>'
              The strategy to encode NA for categorical features.














      ..
          !! processed by numpydoc !!

   .. py:method:: fit(df_train, y_train=None)

      
      Fits NA Encoder.


      :Parameters:

          **df_train** : pandas dataframe of shape = (n_train, n_features)
              The train dataset with numerical and categorical features.

          **y_train** : pandas series of shape = (n_train, ), default = None
              The target for classification or regression tasks.

      :Returns:

          object
              self













      ..
          !! processed by numpydoc !!

   .. py:method:: fit_transform(df_train, y_train=None)

      
      Fits NA Encoder and transforms the dataset.


      :Parameters:

          **df_train** : pandas.Dataframe of shape = (n_train, n_features)
              The train dataset with numerical and categorical features.

          **y_train** : pandas.Series of shape = (n_train, ), default = None
              The target for classification or regression tasks.

      :Returns:

          pandas.Dataframe of shape = (n_train, n_features)
              The train dataset with no missing values.













      ..
          !! processed by numpydoc !!

   .. py:method:: transform(df)

      
      Transform the dataset.


      :Parameters:

          **df** : pandas.Dataframe of shape = (n, n_features)
              The dataset with numerical and categorical features.

      :Returns:

          pandas.Dataframe of shape = (n, n_features)
              The dataset with no missing values.













      ..
          !! processed by numpydoc !!


