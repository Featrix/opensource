Featrix API Usage
=================

.. toctree::

.. autosummary::

1. Account Setup
----------------
Create an account and log in at `https://app.featrix.com`.

2. Generate API Key
-------------------
- Navigate to **API Keys** at the bottom of the left-hand menu.
- Click **Create New API Key**, name your key, and save the **Client ID** and **Client Secret** provided.
  
  .. note::
     These values are not recoverable once the dialog is closed.

3. Logging in with the API
--------------------------
- Install via pip:

.. code-block:: console

   $ pip install featrixclient     # or pip3 install featrixclient

- Instantiate a :class:`Featrix` object in Python to authenticate and connect to the API.

.. code-block:: python

    import featrixclient as ft                                        # pip3 install featrixclient

    FEATRIX_CLIENT_ID = os.environ.get('FEATRIX_CLIENT_ID')           # Put your secrets in a secrets manager!
    FEATRIX_CLIENT_SECRET = os.environ.get('FEATRIX_CLIENT_SECRET') 

    featrix = ft.new_client(client_id=FEATRIX_CLIENT_ID, client_secret=FEATRIX_CLIENT_SECRET)

- You can use :func:`help()` on Python objects to view their docstrings.

Featrix API Overview
=====================

The API provides a primary interface, `Featrix`, and six key objects: `FeatrixUpload`, `FeatrixProject`, `FeatrixEmbeddingSpace`, `FeatrixModel`, and `FeatrixJob`.

- **`Featrix`**: Main entry point for accessing everything in the Featrix environment.
- **`FeatrixUpload`**: Represents data files uploaded to the system. Contains metadata after processing and enrichment.
- **`FeatrixProject`**: Manages embedding spaces, associates uploaded files, sets parameters, and links predictive models to projects.
- **`FeatrixEmbeddingSpace`**: Trained on project data, provides metadata interface for the trained embedding space.
- **`FeatrixNeuralFunction`**: Represents a predictive model. Trained on embedding spaces and specific datasets.
- **`FeatrixJob`**: Represents asynchronous tasks like training, providing progress updates.

The API supports scalar predictions, classifications, recommendations, clustering using embedding spaces. You can use our embeddings with popular vector databases for similarity search of your tabular data. You can also mix in embeddings from any LLM model into Featrix and, by default, Featrix uses sentence-transformer models for string embeddings found in your data.

The API integrates seamlessly with Python libraries like Pandas, Matplotlib, Sklearn, Numpy, and PyTorch. For support or enhancements, contact hello@featrix.ai or join our `Slack community <https://join.slack.com/t/featrixcommunity/shared_invite/zt-25ad0tj5j-3VyaO3YdI8qI4kdr2VhUGA>`_.

.. autoclass::  featrixclient.networkclient.Featrix
    :members:

.. autoclass::  featrixclient.FeatrixUpload
    :members:

.. autoclass::  featrixclient.FeatrixProject
    :members:

.. autoclass::  featrixclient.FeatrixEmbeddingSpace
    :members:

.. autoclass::  featrixclient.FeatrixNeuralFunction
    :members:

.. autoclass::  featrixclient.FeatrixJob
    :members:
