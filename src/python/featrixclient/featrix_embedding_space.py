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

import uuid
from datetime import datetime
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import Field

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .featrix_model import FeatrixModel
from .models import EmbeddingDistanceResponse, ESCreateArgs
from .models import EmbeddingSpace
from .models import PydanticObjectId
from .utils import display_message


class FeatrixEmbeddingSpace(EmbeddingSpace):
    client: Optional[Any] = None
    models_cache: Dict[str, FeatrixModel] = Field(default_factory=dict, exclude=True)
    models_cache_updated: Optional[datetime] = Field(default=None, exclude=True)
    explorer_data: Dict = Field(default=None, exclude=True)

    @staticmethod
    def new_embedding_space(
            fc: Any,
            name: Optional[str] = None,
            credit_budget: int = 3,
            wait_for_completion: bool = False,
            **kwargs
    ):  # noqa
        """
        This creates a chained-job to do training first on an embedding space, and then on the predictive model
        within that embedding space.  It returns a tuple which is the two jobs (the first job for the embedding space
        training and the second for the predictive model training).
        """
        from .featrix_job import FeatrixJob

        if name is None:
            name = f"{fc.current_project.name}-{uuid.uuid4}"

        es_create_args = ESCreateArgs(
            project_id=str(fc.current_project.id),
            name=name,
            training_budget_credits=credit_budget,
        )
        for k, v in kwargs.items():
            if k in ESCreateArgs.__annotations__:
                setattr(es_create_args, k, v)

        dispatch = fc.api.op("job_es_create", es_create_args)
        job = FeatrixJob.from_job_dispatch(dispatch, fc)
        if wait_for_completion:
            while job.finished is False:
                time.sleep(5)
                job = job.check()
                display_message(
                    "Training: {job.incremental_status.message if job.incremental_status is not None else ''}"
                )
            if job.error:
                raise FeatrixException(f"Failed to train embedding space {job.embedding_space_id}: {job.error_msg}")
        es = FeatrixEmbeddingSpace.by_id(job.embedding_space_id, fc)
        return es, job

    @classmethod
    def all(cls, fc) -> List["FeatrixEmbeddingSpace"]:
        results = fc.api.op("es_get_all")
        return ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=fc)

    @classmethod
    def by_id(cls, es_id: PydanticObjectId | str, fc) -> "FeatrixEmbeddingSpace":
        results = fc.api.op("es_get", embedding_space_id=str(es_id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=fc)

    def get_explorer_data(self, force: bool = False) -> Dict:
        if self.explorer_data is None or force:
            self.explorer_data = self.fc.api.op("es_get_explorer", embedding_space_id=str(self.id))
        return self.explorer_data

    def find_training_jobs(self) -> List["FeatrixJob"]:  #noqa
        """
        Find any/all training jobs for this Embedding Space, returning them as a list in order they were executed

        """
        from .featrix_job import FeatrixJob

        results = self.fc.api.op("es_get_training_jobs", embedding_space_id=str(self.id))
        return ApiInfo.reclass(FeatrixJob, results, fc=self.fc)

    def models(self, force: bool = False):
        since = None
        if self.models_cache_updated is not None and not force:
            since = self.models_cache_updated
            result = self.fc.check_updates(model=self.models_cache_updated)
            if result.model is False:
                return list(self.models_cache.values())
        self.models_cache_updated = datetime.utcnow()
        results = self.fc.api.op(
            "es_get_models", embedding_space_id=str(self._id), since=since
        )
        models = ApiInfo.reclass(FeatrixModel, results, fc=self.fc)
        for model in models:
            # FIXME: The model isn't getting serialized correctly
            if hasattr(model, "_id"):
                model.id = getattr(model, "_id")
            self.models_cache[str(model.id)] = model
        return models

    def model(
        self, model_id: str | PydanticObjectId, force: bool = False
    ) -> FeatrixModel:
        model_id = str(model_id)
        if not force:
            if model_id in self.models_cache:
                return self.models_cache[model_id]
        result = self.fc.api.op(
            "es_get_model", embedding_space_id=str(self.id), model_id=str(model_id)
        )
        model = ApiInfo.reclass(FeatrixModel, result, fc=self.fc)
        self.models_cache[model_id] = model
        return model

    def histogram(self) -> EmbeddingDistanceResponse:
        results = self.fc.api.op("es_get_histogram", embedding_space_id=str(self.id))
        return results

    def distance(self) -> EmbeddingDistanceResponse:
        results = self.fc.api.op("es_get_distance", embedding_space_id=str(self.id))
        return results

    def delete(self) -> "FeatrixEmbeddingSpace":
        result = self.fc.api.op("es_delete", embedding_space_id=str(self.id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, result, fc=self.fc)
