Welcome to Featrix!
===================

`Featrix <https://featrix.ai/>`_ is a data gateway to create ML models for structured data *with no data preparation
on your part*. Featrix has both a modern user interface `hosted at app.featrix.com <https://app.featrix.com/>`_ and a Python client library
that can easily be used in a notebook. Featrix can be used by developers, ML engineers, or data scientists to work with any
structured data source, including Pandas dataframes, spreadsheets, databases, or CSV files. You can also mix structured and unstructured data, e.g. Featrix handles plain text fields with no additional work.

Getting started is easy and involves just a few steps:

1. Load your baseline training data into a Featrix project.
2. Wait for a Featrix foundation model to finish training on your data.
3. Choose a target variable (either in the original file, or a separate file with labels), and train a Featrix "neural funtion" to compute predictions.
4. Run predictions on queries (including those with only partial information) using your neural function.
5. (optional) Train additional neural functions for different features/columns using the same foundational model.

A few notes on why we have picked these abstractions:

1. The project lets you mix and match source data into different configurations or arrangements without having to reload the data.
2. Manually joining data is not required to associate data in the project.
3. Featrix trains an embedding space using the project data sources.  These embeddings can be used to create neural functions - prediction models for different features in the data.

A Quick Example
---------------

.. code-block:: python
    #
    # Install the Featrix packet from a command shell with pip or conda, depending on your environment:
    #     pip install featrixclient
    #     or
    #     conda install featrixclient
    # or use the "!" command in a Jupyter notebook cell:
    #     !pip install featrixclient
    #
    import featrixclient as ft

    # You will need a set of API Keys (client id and client secret)
    # generated from https://app.featrix.com. Put these in your environment
    # as FEATRIX_CLIENT_ID and FEATRIX_CLIENT_SECRET or insert below.

    fc = ft.new_client()
    # fc = ft.new_client(client_id="xxxxx", client_secret="yyyyy")


    # If you have a CSV in /home/user/sales.csv which has your sales
    # records including product_type and revenue fields, we can create
    # a neural function that allows you to predict revenue based on
    # product_type (or any other field)
    # Check out more examples on https://www.featrix.ai/

    nf, _, _ = fc.create_neural_function(
            target_fields="revenue",
            project="predict_revenue",
            files=["/home/user/sales.csv"],
            wait_for_completion=True
    )
    prediction = nf.predict(dict(product_type="macbook"))
    print(prediction)
    # [{'revenue': 3255.10}]

Models trained with Featrix can be used to classify data, to make a recommendation, or perform regression. The queries can be new data not seen before by the embedding space; it can also include columns that are not present in the embedding space - although they won't be used to make the prediction, and will just be ignored.


What can Featrix do?
--------------------
We want to make Featrix the easiest way to
    1. explore a data set and uncover non-trivial relationships in it
    2. build world-class predictive AI models for all developers

We want Featrix to enable any developer of modest skill level to build AI applications for classification, regression, recommendation, and clustering. And we want Featrix to have enough power and knobs for even the most sophisticated teams.

We believe vector-based computing is the future and everything we do in Featrix is powered by vector-based embeddings that represent the original data. 


Data Linking
------------
One of the best ways to improve machine learning models is to add new data sources to it. Often the real world
represented by the data has additional context that is missing from our narrow data sources: perhaps our car sales
correspond to weather data or economic sentiment.

Unfortunately, bringing in new data sources into both ML experiments and ML production environments is a
tremendous amount of work. The linking of data records is fraught with dozens of choices for how to link,
aggregate, and pick the data.

Featrix tremendously speeds up this process.

Featrix enables you to bring together data *without having to link it* yourself--so Featrix can sort out the details
for you. Whether you're doing 1-to-1 or 1-to-many or many-to-many associations, Featrix lets you construct data
representations that are robust and leverage your data without your engineering effort required getting bogged
down in tremendous details.

We have tested this linking on many real-world data sets in a wide variety of customer projects and continue
to improve the capabilities.


Data Enrichment: Time and scalars
---------------------------------

Traditional methods of dealing with time data have been somewhat unsatisfactory. Featrix is here to help.

When Featrix detects timestamps, dates, or time strings in a column, it will automatically add new columns capturing
different representations of the time. We add only new columns that are interesting to the data. For example,
if all your times are at the top of the hour, we will not add a minute field.

We do this for strings that appear to include a time description as well; if you have a field that is formatted
like "2 hours", "43 minutes", etc, Featrix will automatically create new columns you can leverage and predict on,
without having to do any of this work yourself.



How Featrix Works
-----------------

Featrix works by creating an embedding space and encoding your data into embeddings in the embedding space.
You can train multiple downstream models within a single embedding space; you can encode data from multiple
sources into a single embedding space, and you can further fine-tune the models for specific tasks downstream.

Featrix gives you an out-of-the-box ML-ready platform to build applications with minimal data preparation
overhead, so you can quickly explore the predictive value of new data sets, operate ML at scale, and enable
statistically rich models with minimal human work in the loop for both test and production environments.


.. note::

    Featrix is currently in private beta with select customers.

    Sign up for our waitlist at `featrix.ai <https://featrix.ai/>`_.

Contents
--------

.. toctree::

   usage
   api
   UNTIE

