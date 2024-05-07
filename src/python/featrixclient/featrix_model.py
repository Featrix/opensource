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

from pydantic import Field

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .featrix_job import FeatrixJob
from .featrix_predictions import FeatrixPrediction
from .models import Model
from .models import ModelFastPredictionArgs
from .models import NewNeuralFunctionArgs
from .models import ModelCreateArgs
from .models import ESCreateArgs


class FeatrixModel(Model):
    fc: Optional[Any] = Field(default=None, exclude=True)
    predictions_cache: Optional[Any] = Field(default_factory=dict, exclude=True)
    predictions_cache_updated: Optional[datetime] = Field(default=None, exclude=True)

    @staticmethod
    def create_args(project_id: str, **kwargs):
        model_create_args = ModelCreateArgs(
            project_id=project_id,
        )
        for k, v in kwargs.items():
            if k in ModelCreateArgs.__annotations__:
                setattr(model_create_args, k, v)
        return model_create_args

    @classmethod
    def by_id(cls, model_id: str, fc: Any):
        result = fc.api.op("model_get", model_id=model_id)
        return  ApiInfo.reclass(cls, result, fc=fc)

    @classmethod
    def from_job(cls, job: FeatrixJob, fc):
        return cls.by_id(str(job.model_id), fc=fc)

    @staticmethod
    def new_neural_function(
            fc: Any,
            target_field: str | List[str],
            credit_budget: int = 3,
            embedding_space: Optional["FeatrixEmbeddingSpace" | str] = None,  # noqa F821 forward ref
            **kwargs
        ):  # noqa
        """
        This creates a chained-job to do training first on an embedding space, and then on the predictive model
        within that embedding space.  It returns a tuple which is the two jobs (the first job for the embedding space
        training and the second for the predictive model training).
        """
        from .featrix_job import FeatrixJob
        from .featrix_embedding_space import FeatrixEmbeddingSpace

        if embedding_space is not None:
            raise NotImplementedError("Not implemented yet")
        if isinstance(target_field, str):
            target_field = [target_field]

        neural_request = NewNeuralFunctionArgs(
            project_id=fc.current_project.id,
            training_credits_budgeted=credit_budget,
            embedding_space_create=FeatrixEmbeddingSpace.create_args(
                str(fc.current_project.id),
                kwargs.get('name', f"Predict_{'_'.join(target_field)}"),
                **kwargs
            ),
            model_create=FeatrixModel.create_args(str(fc.current_project.id), target_columns=target_field, **kwargs)
        )
        dispatches = fc.api.op("job_chained_new_neural_function", neural_request)
        jobs = [FeatrixJob.from_job_dispatch(dispatch, fc) for dispatch in dispatches]
        if len(jobs) == 1:
            # If there was already an embedding space, get the last training job for that ES and return it.
            job_list = FeatrixEmbeddingSpace.by_id(str(jobs[0].embeddingspace_id), fc)
            jobs.insert(0, job_list[-1])
        return jobs[0], jobs[1]

    def predict(self, query: Dict | List[Dict], wait_for_job: bool = True):
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
        pred = self.fc.api.op("models_create_prediction", predict_args)
        fp = ApiInfo.reclass(FeatrixPrediction, pred, fc=self.fc)
        self.predictions_cache[fp.id] = fp
        if self.fc.debug:
            print(f"Prediction {fp.id}: {fp.query} -> {fp.result}")
            print(
                f"Prediction timing: Overall {fp.debug_info.get('api_time', -1)} "
                f"Query {fp.debug_info.get('prediction_time', -1)}"
            )
        return fp.result

    def predictions(self, force: bool = False):
        """
        Retrieve historical predictions.
        """
        from .featrix_predictions import FeatrixPrediction

        since = None
        if self.predictions_cache_updated is not None and not force:
            since = self.predictions_cache_updated
            result = self.fc.check_updates(job_meta=self.predictions_cache_updated)
            if result.job_meta is False:
                return list(self.predictions_cache.values())
        self.predictions_cache_updated = datetime.utcnow()
        results = self.fc.api.op(
            "models_get_predictions", model_id=str(self.id), since=since
        )
        prediction_list = ApiInfo.reclass(FeatrixPrediction, results, fc=self.fc)
        for prediction in prediction_list:
            self.predictions_cache[str(prediction.id)] = prediction
        return prediction_list

    def get_prediction(self, prediction_id: str) -> FeatrixPrediction:
        return self.predictions_cache.get(str(prediction_id))

    def find_prediction(self, **kwargs) -> Tuple[FeatrixPrediction, Dict]:
        """
        Given a set of column to value's that make up a query, look to see if that query is contained in
        any of our predictions -- and if so, return it. Otherwise, return None.
        """
        for prediction in self.predictions_cache.values():
            match, idx = prediction.match(**kwargs)
            if match:
                return prediction, prediction.result[idx]
        raise FeatrixException("No prediction found for query")
