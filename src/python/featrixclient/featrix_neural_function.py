#  -*- coding: utf-8 -*-
#############################################################################
#
#  Copyright (c) 2024, Featrix, Inc. All rights reserved.
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
#
#
#############################################################################
#
#  Yes, you can see this file, but Featrix, Inc. retains all rights.
#
#############################################################################
from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import Field, PrivateAttr

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .featrix_job import FeatrixJob
from .featrix_predictions import FeatrixPrediction
from .models import Model, JobType
from .models import ModelFastPredictionArgs
from .models import NewNeuralFunctionArgs
from .models import ModelCreateArgs
from .models import ESCreateArgs
from .config import settings


class FeatrixNeuralFunction(Model):
    """
    Represents a predictive model, also known as a neural function.  It is a model trained against an embedding space
    to provide prediction for a given feature/field.

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
    def by_id(cls, model_id: str, fc: Any) -> "FeatrixNeuralFunction":
        """
        Get a predictive model by its id.

        Returns:
            FeatrixNeuralFunction: The model if it exists, otherwise None.
        """
        result = fc.api.op("model_get", model_id=model_id)
        return  ApiInfo.reclass(cls, result, fc=fc)

    @classmethod
    def from_job(cls, job: FeatrixJob, fc):
        """
        Given a FeatrixJob, retrieve the predictive model that is referenced by the job.

        Arguments:
            job: FeatrixJob: The job that references the model -- must have model_id set

        Returns:
            FeatrixNeuralFunction: The model if it exists, otherwise None.
        """
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


    @staticmethod
    def new_neural_function(
            fc: Any,
            project: "FeatrixProject" | str,  # noqa F821 forward ref
            target_field: str | List[str],
            credit_budget: int = 3,
            encoder: Optional[Dict] = None,
            ignore_cols: Optional[List[str] | str] = None,
            focus_cols: Optional[List[str] | str] = None,
            embedding_space: Optional["FeatrixEmbeddingSpace" | str] = None,  # noqa F821 forward ref
            **kwargs
        ):
        """
        This creates a chained-job to do training first on an embedding space, and then on the predictive model
        within that embedding space.  It returns a tuple which is the two jobs (the first job for the embedding space
        training and the second for the predictive model training).

        Arguments:
            fc: the feature client object for submitting API requests
            target_field: The field that we are trying to predict
            credit_budget: The number of credits to budget for training
            embedding_space: The embedding space to use for training
            encoder: Optional encoder overrides to use for training the embedding space
            ignore_cols: Optional columns to ignore during training
            focus_cols: Optional columns to focus on during training
            kwargs: Additional arguments to pass to the create embedding or train model functions

        """
        from .featrix_job import FeatrixJob
        from .featrix_project import FeatrixProject
        from .featrix_embedding_space import FeatrixEmbeddingSpace

        project_id = str(project.id) if isinstance(project, FeatrixProject) else project
        embedding_space_create = FeatrixEmbeddingSpace.create_args(
            project_id,
            kwargs.get('name', f"Predict_{'_'.join(target_field)}"),
            encoder=encoder or {},
            ignore_cols=ignore_cols or [],
            focus_cols=focus_cols or [],
            **kwargs
        )
        if embedding_space is not None:
            embedding_space_create.embedding_space_id = str(embedding_space.id)

        if isinstance(target_field, str):
            target_field = [target_field]

        neural_request = NewNeuralFunctionArgs(
            project_id=project_id,
            training_credits_budgeted=credit_budget,
            embedding_space_create=embedding_space_create,
            model_create=FeatrixNeuralFunction.create_args(project_id, target_columns=target_field, **kwargs)
        )
        dispatches = fc.api.op("job_chained_new_neural_function", neural_request)
        jobs = [FeatrixJob.from_job_dispatch(dispatch, fc) for dispatch in dispatches]
        if len(jobs) == 1:
            # If there was already an embedding space, get the last training job for that ES and return it.
            job_list = FeatrixEmbeddingSpace.by_id(str(jobs[0].embeddingspace_id), fc)
            jobs.insert(0, job_list[-1])
        return jobs[0], jobs[1]

    def get_jobs(self, active: bool = True, training: bool = True) -> List[FeatrixJob]:
        """
        Return a list of jobs that are associated with this model.  By default it will only
        return active (not finished) training jobs, but the caller can use the two arguments to control this.

        Arguments:
            active: bool: If True, only return active jobs
            training: bool: If True, only return training jobs

        Returns:
            List[FeatrixJob]: The list of jobs associated with this model
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

    def predictions(self, stale_timeout: int = settings.stale_timeout) -> List[FeatrixPrediction]:
        """
        Retrieve historical predictions.

        Arguments:
            stale_timeout: int: The number of seconds to wait before refreshing the cache

        Returns:
            List[FeatrixPrediction]: The list of predictions for this model
        """
        from .featrix_predictions import FeatrixPrediction

        if (
                self._predictions_cache_updated is None or
                (datetime.utcnow() - self._predictions_cache_updated) > stale_timeout
        ):
            since = self._predictions_cache_updated
            self._predictions_cache_updated = datetime.utcnow()
            results = self._fc.api.op(
                "models_get_predictions", model_id=str(self.id), since=since
            )
            prediction_list = ApiInfo.reclass(FeatrixPrediction, results, fc=self._fc)
            for prediction in prediction_list:
                self._predictions_cache[str(prediction.id)] = prediction
        return list(self._predictions_cache.values())

    def prediction(self, prediction_id: str, stale_timeout: int = settings.stale_timeout) -> FeatrixPrediction:
        """
        Get a prediction by its id from the cache.
        """
        if prediction_id not in self._predictions_cache:
            self.predictions(stale_timeout=stale_timeout)
        if prediction_id in self._predictions_cache:
            return self._predictions_cache[prediction_id]
        raise RuntimeError(f"Prediction {prediction_id} not found in Neural Function {self.id}")

    def find_prediction(self, **kwargs) -> Tuple[FeatrixPrediction, Dict]:
        """
        Given a set of column to value's that make up a query, look to see if that query is contained in
        any of our predictions -- and if so, return it. Otherwise, return None.
        """
        for prediction in self._predictions_cache.values():
            match, idx = prediction.match(**kwargs)
            if match:
                return prediction, prediction.result[idx]
        raise FeatrixException("No prediction found for query")
