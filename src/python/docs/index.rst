Welcome to Featrix!
===================

`Featrix <https://featrix.ai/>`_ is a data gateway to create ML models for structured data *with no data preparation on your part*. Featrix comes with a Python client library for ML engineers and data scientists to work with any structured data source, including Pandas dataframes. Featrix is powered by a hosted SaaS or private Docker containers deployed on site in private clouds with an enterprise license.

Getting started is easy and involves just a few steps:

1. Load your baseline training data into a Featrix "data space".
2. Train a "vector space" on that data. This transforms the original data into vectors that you can leverage for models or querying.
3. At this point, you can cluster the vectors or query for nearest neighbors with no further work.
4. You can also train a downstream prediction model for a target column. The target column can be in the original data, or it can be something specific to the model itself.
5. Then you can run the model. The model can be presented with partial records and it returns values for the target.

A few notes on why we have picked these abstractions:

1. The data space lets you mix and match source data into different configurations or arrangements without having to reload the data.
2. Manually joining data is not required to associate data in the data space; Featrix infers likely combinations to associate data and you can choose to override these if needed.
3. A data space can have multiple vector spaces with different arrangements.
4. A vector space can have multiple models.
5. Every vector space includes a set of vector indices to enable extremely fast querying for clusters or nearest neighbors in the data set.

A Quick Example
---------------

.. code-block:: python

    import featrixclient as ft

    dataspace_name = "new-prospects"

    featrix = ft.Featrix("http://localhost:8080")
    if not featrix.EZ_DataSpaceExists(dataspace_name):
        dataspace = featrix.EZ_DataSpaceCreate(dataspace_name)

    # Load some data
    featrix.EZ_DataSpaceLoadIfNeeded(dataspace_name, "prospect-list.csv",     "biglist")
    featrix.EZ_DataSpaceLoadIfNeeded(dataspace_name, "had-meetings-list.csv", "meetings")

    # Link the data
    links = featrix.EZ_DataAutoJoin(dataspace_name)
    print("Links:", links)

    # Train a vector space
    vector_space_id = featrix.EZ_NewVectorSpace(dataspace=dataspace_name)

    # At this point, we can create a model or query the data for similar neighbors and so on.


Models trained with Featrix can be used to classify data, to make a recommendation, or perform a regression on some data. The data can be new data not seen before by the vector space; it can also include columns that are not present in the vecotr space.


What can Featrix do?
--------------------
We have big goals for Featrix, including:

1. Speed and ease of development.
2. Flexible enough for power uses under the hood, but a set of easy to use APIs and tools that are simple enough for novice organizations to get started with ML right away.
3. Enable a variety of downstream workloads with the Featrix data representations. We believe vector-based computing is the future and everything we do in Featrix is powered by vector-based embeddings that represent the original data.


Data Linking
------------
One of the best ways to improve machine learning models is to add new data sources to it. Often the real world represented by the data has additional context that is missing from our narrow data sources: perhaps our car sales correspond to weather data or economic sentiment.

Unfortunately, bringing in new data sources into both ML experiments and ML production environments is a tremendous amount of work. The linking of data records is fraught with dozens of choices for how to link, aggregate, and pick the data.

Featrix tremendously speeds up this process.

Featrix enables you to bring together data *without having to link it* yourself--so Featrix can sort out the details for you. Whether you're doing 1-to-1 or 1-to-many or many-to-many associations, Featrix lets you construct data representations that are robust and leverage your data without your engineering effort requirred getting bogged down in tremendous details.

We have tested this linking on many real-world data sets in a wide variety of customer projects and continue to improve the capabilities.

.. code-block:: python

    # What's returned by auto join
    links = featrix.EZ_DataAutoJoin(dataspace_name)

    print(links)


Data Enrichment: Time and scalars
---------------------------------

Traditional methods of dealing with time data have been somewhat unsatisfactory. Featrix is here to help.

When Featrix detects timestamps, dates, or time strings in a column, it will automatically add new columns capturing different representations of the time. We add only new columns that are interesting to the data. For example, if all your times are at the top of the hour, we will not add a minute field.

We do this for strings that appear to include a time description as well; if you have a field that is formatted like "2 hours", "43 minutes", etc, Featrix will automatically create new columns you can leverage and predict on, without having to do any of this work yourself.



How Featrix Works
-----------------

Featrix works by creating an embedding space and encoding your data into embeddings in the embedding space. You can train multiple downstream models within a single embedding space; you can encode data from multiple sources into a single embedding space, and you can further fine-tune the models for specific tasks downstream.

Featrix gives you an out-of-the-box ML-ready platform to build applications with minimal data preparation overhead, so you can quickly explore the predictive value of new data sets, operate ML at scale, and enable statistically rich models with minimal human work in the loop for both test and production environments.




.. note::

    Featrix is currently in private beta with select customers.

    Sign up for our waitlist at `featrix.ai <https://featrix.ai/>`_.

Contents
--------

.. toctree::

   usage
   api
   UNTIE

