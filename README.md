# featrixclient
     _______ _______ _______ _______ ______ _______ ___ ___
    |    ___|    ___|   _   |_     _|   __ \_     _|   |   |
    |    ___|    ___|       | |   | |      <_|   |_|-     -|
    |___|   |_______|___|___| |___| |___|__|_______|___|___|

Featrix is a data gateway to create ML models for structured data with no data preparation on your part. Featrix comes with a Python client library for ML engineers and data scientists to work with any structured data source, including Pandas dataframes. 

## Getting started is easy and involves just a few steps:

1. Create an account at [app.featrix.com](https://app.featrix.com/)
2. Load your baseline training data into a Featrix project.
3. Train an “embedding space” on that data. The embedding space is a trained neural network that creates vector embeddings from data that is passed in a format like your original data.
4. At this point, you can cluster the embeddings or query for nearest neighbors with no further work.
5. You can also train a downstream prediction model for a target column. The target column can be in the original data, or it can be something specific to the model itself.
6. Then you can run the model. The model can be presented with partial records and it returns values for the target.

## Why these abstractions?

The project lets you mix and match source data into different configurations or arrangements without having to reload the data. Data belongs to the team (organization) that held ownership on the data.

Manually joining data is not required to associate data in the data space; Featrix infers likely combinations to associate data and you can choose to override these if needed.

An embedding space can have multiple models.

We have videos and full docs up at [www.featrix.ai](https://www.featrix.ai/).

### Questions?

Drop us a note at support@featrix.ai
