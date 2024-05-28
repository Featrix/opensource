Featrix Client
==============

.. toctree::

.. autosummary::

To use Featrix, you will need an account on `https://app.featrix.com`.  You can preform all operations via the web interface, or you can generate an API key on the web interface and
use that API key to interact with Featrix programmatically via the object API provided in Python.  This documentation describes the object API interface.

==============
Authentication
==============

To use the API, you will need to create an account and login to `https://app.featrix.com`.  Once logged in, select the *API Keys* item near the bottom of the left-hand navigation pane.   This will take you *Manage API Keys* screen, where you create a new key by selecting the *Create New Api Key* button and providing a name for your key.
Working with it is pretty easy. You can call help() on Python objects to see the docstrings on the objects.

Generally, you create a Featrix object, which will connect to the Featrix server, perform authentication and give you an object to interact with the API easily.

In order to use the API, you will need an API key from the web interface.  Once you have created an account and logged into `https://app.featrix.com`, you can select the *Api Keys* menu item on the bottom of the left navigation pane, , which consist of a client id and client secret, which the web interface will give you when you generate the API key. This will display a dialog
box wiht the *Client ID* and *Client Secret* that you will need to use the API.  You need to save these values in a secure place, as they are not recoverable once you dismiss the dialog.  You can provide these keys to the API in multiple ways, as discussed below.

===
API
===

The API contains a primary interface (`Featrix`) and six primary objects: `FeatrixUpload`, `FeatrixProject`, `FeatrixEmbeddingSpace`, `FeatrixModel`, `FeatrixPrediction`, and `FeatrixJob`.

The `Featrix` has most of the major entry points for uploading files, creating projects, training embedding spaces, neural functions/models. These interfaces return one or more of the primary objects, and each of those provides an additional set of methods.

The `FeatrixUpload` object is used to represent data files that up upload into the system.  They are processed and will contain meta-data about what Featrix discovered as it interrogated and enriched the file.

The `FeatrixProject` is where you create your embedding spaces. It allows you to associate files you have uploaded with the project, and set parameters for creating embedding spaces.  Additionally, predictive models (neural functions), which are created from embeddings, are associated with the project as well.

The `FeatrixEmbeddingSpace` is trained on the data files associated with a project and provides the metadata interface to your trained Embedding Space.

The `FeatrixModel` (or Neural Function) represents a predictive model. It is trained using an embedding space and either new data for the model specifically or some of the data that the embedding space was trained on, depending on your application needs.

The `FeatrixPrediction` represents a prediction you have run against a neural function/predictive model. It has both the query you used as well as the result from that query and allows you to retrieve the results of previous predictions.

The `FeatrixJob` represents asynchronous jobs that are being run on your behalf such as training an embedding space or model. They also provide incremental feedback about the job's progress.


You can build scalar predictions, classifications, recommendations, and more with this API. You can also cluster data or query for nearest neighbors by leveraging the embedding space. You can extend the embedding spaces, branch them, tune their training, and more.

We have designed this API to work with standard Python ecosystems: you should be able to easily connect Pandas, databases, matplotlib, sklearn, numpy, and PyTorch with the API. If you run into issues or want enhancements, drop us a note at mitch@featrix.ai or join our `Slack community <https://join.slack.com/t/featrixcommunity/shared_invite/zt-25ad0tj5j-3VyaO3YdI8qI4kdr2VhUGA>`!


.. autoclass::  featrixclient.networkclient.Featrix
    :members:

.. autoclass::  featrixclient.FeatrixProject
    :members:

.. autoclass::  featrixclient.FeatrixEmbeddingSpace
    :members:

.. autoclass::  featrixclient.FeatrixModel
    :members:

.. autoclass::  featrixclient.FeatrixPrediction
    :members:

.. autoclass::  featrixclient.FeatrixJob
    :members:
