#  -*- coding: utf-8 -*-
#############################################################################
#
#  Copyright (c) 2024, Featrix, Inc.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#############################################################################
#
#     Welcome to...
#
#      _______ _______ _______ _______ ______ _______ ___ ___
#     |    ___|    ___|   _   |_     _|   __ \_     _|   |   |
#     |    ___|    ___|       | |   | |      <_|   |_|-     -|
#     |___|   |_______|___|___| |___| |___|__|_______|___|___|
#
#                                                 Let's embed!
#
#############################################################################
#
#  Sign up for Featrix at https://app.featrix.com/
# 
#############################################################################
#
#  Check out the docs -- you can either call the python built-in help()
#  or fire up your browser:
#
#     https://featrix-docs.readthedocs.io/en/latest/
#
#  You can also join our community Slack:
#
#     https://join.slack.com/t/featrixcommunity/shared_invite/zt-28b8x6e6o-OVh23Wc_LiCHQgdVeitoZg
#
#  We'd love to hear from you: bugs, features, questions -- send them along!
#
#     hello@featrix.ai
#
#############################################################################
#
from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import bson
from pydantic import PrivateAttr

from .api_urls import ApiInfo
from .config import settings
from .exceptions import FeatrixException
from .featrix_job import FeatrixJob
from .featrix_predictions import FeatrixPrediction
from .models import JobType
from .models import Model
from .models import ModelCreateArgs
from .models import ModelFastPredictionArgs
from .models import NewNeuralFunctionArgs
from .models import PydanticObjectId


class FeatrixNeuralFunction(Model):
    """
    Represents a neural function (predictive model) trained against an embedding space for predicting a specific feature or field.

    Create and train a new neural function with `.new_neural_function()`. Retrieve an existing neural function by its ID with `.by_id()`, and ensure you have the latest version of the model using `.refresh()`:

        nf = FeatrixNeuralFunction.by_id("5f7b1b1b1b1b1b1b1b1b1b")
        nf = nf.refresh()  # Update after training completes

    """

    _fc: Optional[Any] = PrivateAttr(default=None)
    """Reference to the Featrix class  that retrieved or created this project, used for API calls/credentials"""
    _predictions_cache: Optional[Any] = PrivateAttr(default_factory=dict)
    _predictions_cache_updated: Optional[datetime] = PrivateAttr(default=None)

    @staticmethod
    def create_args(project_id: str, **kwargs) -> ModelCreateArgs:
        """
        Create the arguments for creating a model.  This is a helper function to make it easier to create a
        `ModelCreateArgs`
        """
        model_create_args = ModelCreateArgs(
            project_id=project_id,
        )
        for k, v in kwargs.items():
            if k in ModelCreateArgs.__annotations__:
                setattr(model_create_args, k, v)
        return model_create_args

    @classmethod
    def by_id(
        cls, model_id: str | PydanticObjectId, fc: Any
    ) -> "FeatrixNeuralFunction":
        """
        Get a predictive model by its id.

        Returns:
            FeatrixNeuralFunction: The model if it exists, otherwise None.
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()
        result = fc.api.op("model_get", model_id=str(model_id))
        return ApiInfo.reclass(cls, result, fc=fc)

    @classmethod
    def from_job(
        cls, job: FeatrixJob, fc: Optional["Featrix"] = None
    ) -> "FeatrixNeuralFunction":  # noqa F821
        """
        Given a FeatrixJob, retrieve the predictive model that is referenced by the job.

        Arguments:
            job: FeatrixJob: The job that references the model -- must have model_id set

        Returns:
            FeatrixNeuralFunction: The model if it exists, otherwise None.
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()
        return cls.by_id(str(job.model_id), fc=fc)

    @property
    def fc(self):
        return self._fc

    @fc.setter
    def fc(self, value):
        from .networkclient import Featrix

        if isinstance(value, Featrix) is False:
            raise FeatrixException("fc must be an instance of Featrix")

        self._fc = value

    def refresh(self):
        return self.by_id(self.id, self.fc)

    @classmethod
    def new_neural_function(
        cls,
        fc: Any,
        project: "FeatrixProject" | str,  # noqa F821 forward ref
        target_field: str | List[str],
        credit_budget: int = 3,
        encoder: Optional[Dict] = None,
        ignore_cols: Optional[List[str] | str] = None,
        focus_cols: Optional[List[str] | str] = None,
        embedding_space: Optional["FeatrixEmbeddingSpace" | str] = None,  # noqa F821 forward ref
        **kwargs,
    ) -> "FeatrixNeuralFunction":
        """
        Create a chained job to first train an embedding space, then the predictive model within that space. Returns the created `FeatrixNeuralFunction`. Use `embedding_space_id` and `.get_jobs()` to retrieve the embedding space and training jobs.

        Args:
            fc (FeatureClient): The feature client object for API requests.
            target_field (str): The field to predict.
            credit_budget (int): Credits allocated for training.
            embedding_space (EmbeddingSpace): The embedding space to use for training.
            encoder (dict | None): Optional encoder overrides for training.
            ignore_cols (list | str | None): Columns to ignore during training.
            focus_cols (list | str | None): Columns to focus on during training.
            kwargs (dict): Additional arguments for embedding creation or model training.

        Returns:
            FeatrixNeuralFunction: The created predictive model.
        """

        from .featrix_job import FeatrixJob
        from .featrix_project import FeatrixProject
        from .featrix_embedding_space import FeatrixEmbeddingSpace

        project_id = str(project.id) if isinstance(project, FeatrixProject) else project
        if bson.ObjectId.is_valid(project_id) is False:
            raise FeatrixException(
                f"Invalid project id ({project_id}) passed in new_neural_function"
            )
        name = kwargs.pop("name", f"Predict_{'_'.join(target_field)}")
        embedding_space_create = FeatrixEmbeddingSpace.create_args(
            project_id=project_id,
            name=name,
            encoder=encoder or {},
            ignore_cols=ignore_cols or [],
            focus_cols=focus_cols or [],
            **kwargs,
        )
        if embedding_space is not None:
            embedding_space_create.embedding_space_id = str(embedding_space.id)

        if isinstance(target_field, str):
            target_field = [target_field]

        neural_request = NewNeuralFunctionArgs(
            project_id=project_id,
            training_credits_budgeted=credit_budget,
            embedding_space_create=embedding_space_create,
            model_create=FeatrixNeuralFunction.create_args(
                project_id, target_columns=target_field, **kwargs
            ),
        )
        dispatches = fc.api.op("job_chained_new_neural_function", neural_request)
        jobs = [FeatrixJob.from_job_dispatch(dispatch, fc) for dispatch in dispatches]

        # Get the NF from the NF job in the job list.
        for job in jobs:
            if job.job_type == JobType.JOB_TYPE_MODEL_CREATE:
                return cls.by_id(str(job.model_id), fc)
        else:
            raise FeatrixException("No model job found in chained jobs")

    def get_jobs(self, active: bool = True, training: bool = True) -> List[FeatrixJob]:
        """
        Return a list of jobs associated with this model.

        By default, only active (unfinished) training jobs are returned. Use the arguments to filter results.

        Args:
            active (bool): If `True`, return only active jobs.
            training (bool): If `True`, return only training jobs.

        Returns:
            List[FeatrixJob]: The list of jobs associated with this model.
        """

        jobs = []
        for job in FeatrixJob.by_neural_function(self):
            if active or training:
                if active and job.finished:
                    continue
                if training and job.job_type != JobType.JOB_TYPE_MODEL_CREATE:
                    continue
            jobs.append(job)
        return jobs

    def predict(self, query: Dict | List[Dict]) -> FeatrixPrediction:
        """
        Predict a probability distribution on a given model in a embedding space.

        Query can be a list of dictionaries or a dictionary for a single query.

        Parameters
        ----------
        query : dict or [dict]
            Either a single parameter or a list of parameters.
            { col1: <value> }, { col2: <value> }

        Returns
        -------
        A dictionary of values of the model's target_column and the probability of each of those values occurring.

        """
        if isinstance(query, dict):
            query = [query]
        predict_args = ModelFastPredictionArgs(model_id=str(self.id), query=query)
        pred = self._fc.api.op("models_create_prediction", predict_args)
        fp = ApiInfo.reclass(FeatrixPrediction, pred, fc=self._fc)
        self._predictions_cache[fp.id] = fp
        if self._fc.debug:
            print(f"Prediction {fp.id}: {fp.query} -> {fp.result}")
            print(
                f"Prediction timing: Overall {fp.debug_info.get('api_time', -1)} "
                f"Query {fp.debug_info.get('prediction_time', -1)}"
            )
        return fp.result
