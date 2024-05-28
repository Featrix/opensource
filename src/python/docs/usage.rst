Usage
=====

.. meta::
   :description: Using the Featrix client API for creating data embeddings.
   :keywords: featrix, featrixclient, python, pytorch, ml, ai

.. highlight:: python
    :linenothreshold: 3


.. _installation:

Installation
------------

To use Featrix, first install the client using pip:

.. code-block:: console

   $ pip install featrix-client     # Coming soon.


You'll also need a Featrix server; you can run the enterprise edition on-site in your environment or use our hosted SaaS.


What's Included
---------------

The ``featrix-client`` package includes a few key modules:

+-------------------+-----------------------------------------------------------+
| ``networkclient`` | A `FeatrixTransformerClient` for                          |
|                   | accessing a Featrix embedding service.                    |
+-------------------+-----------------------------------------------------------+
| ``graphics``      | A set of functions for plotting embedding similarity.     |
++------------------+-----------------------------------------------------------+
| ``utils``         | A set of functions for working with data that we have     |
|                   | found to be useful.                                       |
+-------------------+-----------------------------------------------------------+

Working with Data
-----------------


.. code-block:: python

    import featrixclient as ft
    import pandas as pd
    df = pd.read_csv(path_to_your_file)

Train a vector space and a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can train multiple models on a single vector space.

Check out our `live Google Colab demo notebooks <https://featrix.ai/demo>` for examples. The general approach is as follows:


.. code-block:: python

    # Split the data
    df_train, df_test = train_test_split(df, test_size=0.25)

    # Connect to the Featrix server. This can be deployed on prem with Docker
    # or Featrix’s public cloud.
    featrix = ft.Featrix("http://embedding.featrix.com:8080")

    # Here we create a new vector space and train it on the data.
    vector_space_id = featrix.EZ_NewVectorSpace(df_train)

    # We can create multiple models within a single vector space.
    # This lets us re-use representations for different predictions
    # without retraining the vector space.
    # Note, too, that you could train the model on a different training
    # set than the vector space, if you want to zero in on something
    # for a specific model.
    model_id = featrix.EZ_NewModel(vector_space_id,
                                   "Target_column",
                                    df_train)

    # Run predictions
    result = featrix.EZ_PredictionOnDataFrame(vector_space_id,
                                              Model_id,
                                              "Target_column",
                                              df_test)

    # Now result is a list of classifications in the same symbols
    # as the target column



Predicting on a probability distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can specify a few characteristics of an object and ask for the target field probability distribution. For example, in our mortgage loan demo, we might ask "what are the chances someone who is married will be approved for a loan?"


    >>> # result_married_only
    >>> featrix.EZ_Prediction(vector_space_id, model_id, {"Married": "Yes"})
    {'<UNKNOWN>': 0.0011746988166123629, 'N': 0.33159884810447693, 'Y': 0.6672264933586121}

We can pass in multiple criteria:

    >>> # result_married_and_not_graduate
    >>> featrix.EZ_Prediction(vector_space_id, model_id, {"Education": "Not Graduate", "Married": "Yes"})
    {'<UNKNOWN>': 0.003182089189067483, 'N': 0.5865148305892944, 'Y': 0.41030314564704895}


Classifying records
^^^^^^^^^^^^^^^^^^^

We can determine a category an object belongs to. Typically we'll pass in a list of objects and get back a vector of which class each object targets. Featrix includes an `EZ_PredictionOnDataFrame` call to facilitate passing objects in bulk.

The interface is similar to sklearn's clf.predict() functions. The target column is specified to ensure it is removed from the query dataframe before passing to the model, if it is present.

    >>> featrix.EZ_PredictionOnDataFrame(vector_space_id,
                                          model_id,
                                          "Loan_Status",        # target column name
                                          query_df)
     ['Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y'
      'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'Y' 'N' 'N'
      'N' 'Y' 'N' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y'
      'Y' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y'
      'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y'
      'Y' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y'
      'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y'
      'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y'
      'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y'
      'Y' 'Y' 'Y' 'Y' 'N' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y'
      'Y' 'Y' 'Y' 'Y' 'Y']


Note that we can use the usual sklearn functions to test accuracy, precision, and recall.

    >>> from sklearn.metrics import precision_score, recall_score, accuracy_score
    >>> result = # query from above
    >>> accuracy_score(df_test_loan_status, result)
    0.827027027027027
    >>> precision_score(df_test_loan_status, result, pos_label="Y")
    0.802547770700637
    >>> recall_score(df_test_loan_status, result, pos_label="Y")
    0.992125984251968


Regression
^^^^^^^^^^

Prediction on a continuous variable works in the same way as a query on a categorical variable.

