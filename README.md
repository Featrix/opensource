# featrixclient

Featrix Client Library

Welcome to Featrix!

Featrix is a data gateway to create ML models for structured data with no data preparation on your part. Featrix comes with a Python client library for ML engineers and data scientists to work with any structured data source, including Pandas dataframes. Featrix is powered by a hosted SaaS or private Docker containers deployed on site in private clouds with an enterprise license.

Getting started is easy and involves just a few steps:

Load your baseline training data into a Featrix “data space”.
Train a “vector space” on that data. This transforms the original data into vectors that you can leverage for models or querying.
At this point, you can cluster the vectors or query for nearest neighbors with no further work.
You can also train a downstream prediction model for a target column. The target column can be in the original data, or it can be something specific to the model itself.
Then you can run the model. The model can be presented with partial records and it returns values for the target.
A few notes on why we have picked these abstractions:

The data space lets you mix and match source data into different configurations or arrangements without having to reload the data.
Manually joining data is not required to associate data in the data space; Featrix infers likely combinations to associate data and you can choose to override these if needed.
A data space can have multiple vector spaces with different arrangements.
A vector space can have multiple models.
Every vector space includes a set of vector indices to enable extremely fast querying for clusters or nearest neighbors in the data set.
