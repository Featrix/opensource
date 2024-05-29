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

from pydantic import Field, PrivateAttr

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .featrix_model import FeatrixModel
from .models import EmbeddingDistanceResponse, ESCreateArgs
from .models import EmbeddingSpace
from .models import PydanticObjectId
from .utils import display_message


class FeatrixEmbeddingSpace(EmbeddingSpace):
    """
        Represents a multimodal embedding space.

        This lets you create multimodal embeddings for assorted tabular data sources in a single
        trained embedding space, building a foundational model on your data
        in such a way that you can query the entire data by using partial information that maps into
        as little as one of your original data sources, or you can leverage partial information
        spanning multiple data sources.

        You create embedding spaces in a project, and the training uses data files you have associated with
        the project in the training. You can use a subset of the data,
        ignore columns, or change mappings to rapidly experiment with models using this call.

        This function will use auto-join to find the linkage and corresponding overlapping mutual information between
        data files that have been loaded. Then a new embedding space is trained with the following columns:

        * Base data file: all columns (unless ignored in the ignore_cols parameter)
        * 2nd data file:  all columns, renamed to <2nd data file label> + "_" + <original_col_name>
                        However, the columns used for linking will not be present, as they
                        will get their mapped names in the base data file.

                        To ignore a column in the 2nd data file, specify the name in the
                        transformed format.
        * 3rd data file:  same as 2nd data file.

        This trains the embedding space in the following manner:

            Let's imagine the 2nd_file_col1 and 3rd_file_col2 are the linkage to col1 in the base
            data set. The training space will effectively be a sparse matrix:

        .. code-block:: text

                col1                    col2          col3        2nd_file_col2       2nd_file_col3       3rd_file_col2
                values from base data.....................        [nulls]                                 [nulls]
                .
                .
                .
                2nd_file_col1 in col1   [nulls]                   values from 2nd file................... [nulls]
                .                       .                         .
                .                       .                         .
                3rd_file_col1 in col2   [nulls]                   [nulls]                                 values from 3rd file
                .                       .                         .                                       .
                .                       .                         .                                       .
                .                       .                         .                                       .

    """
    fc: Optional[Any] = None
    """Reference to the Featrix class that retrieved or created this project, used for API calls/credentials"""
    _models_cache: Dict[str, FeatrixModel] = PrivateAttr(default_factory=dict)
    _models_cache_updated: Optional[datetime] = PrivateAttr(default=None)
    _explorer_data: Dict = PrivateAttr(default=None)

    @staticmethod
    def create_args(project_id: str, name: str, **kwargs) -> ESCreateArgs:
        """
        Create the arguments for an embedding space creation call with the parameters passed in.

        Returns an ESCreateArgs object that can be passed to the API for creating an embedding space.
        """
        es_create_args = ESCreateArgs(
            project_id=project_id,
            name=name,
        )
        for k, v in kwargs.items():
            if k in ESCreateArgs.__annotations__:
                setattr(es_create_args, k, v)
        return es_create_args

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

        es_create_args = FeatrixEmbeddingSpace.create_args(
            str(fc.current_project.id),
            name,
            training_budget_credits=credit_budget,
            **kwargs
        )

        dispatches = fc.api.op("job_es_create", es_create_args)
        jobs = [FeatrixJob.from_job_dispatch(dispatch, fc) for dispatch in dispatches]

        if wait_for_completion:
            for job in jobs:
                while job.finished is False:
                    time.sleep(5)
                    job = job.check()
                    display_message(
                        f"{job.job_type}: {job.incremental_status.message if job.incremental_status is not None else ''}"
                    )
                if job.error:
                    raise FeatrixException(f"Failed to train embedding space {job.embedding_space_id}: {job.error_msg}")
        es = FeatrixEmbeddingSpace.by_id(jobs[-1].embedding_space_id, fc)
        es.fc = fc
        return es, jobs[0]

    @classmethod
    def all(cls, fc) -> List["FeatrixEmbeddingSpace"]:
        """
        Return a list of all embedding spaces defined by the user (regardless of project)

        Args:
            fc: Featrix class instance

        Returns:
            List of FeatrixEmbeddingSpace objects
        """
        results = fc.api.op("es_get_all")
        return ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=fc)

    @classmethod
    def by_id(cls, es_id: PydanticObjectId | str, fc) -> "FeatrixEmbeddingSpace":
        """
        Retrieve an embedding space by its ID from the server

        Arguments:
            es_id: The ID of the embedding space
            fc: Featrix class instance

        Returns:
            FeatrixEmbeddingSpace object
        """
        results = fc.api.op("es_get", embedding_space_id=str(es_id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=fc)

    def get_explorer_data(self, force: bool = False) -> Dict:
        """
        Retrieve the explorer data for this embedding space, which includes the neural attributes

        Args:
            force: If True, force a refresh of the explorer data

        Returns:
            Dict of the explorer data
        """
        if self._explorer_data is None or force:
            self._explorer_data = self.fc.api.op("es_get_explorer", embedding_space_id=str(self.id))
        return self._explorer_data

    def find_training_jobs(self) -> List["FeatrixJob"]:  # noqa forward ref
        """
        Find any/all training jobs for this Embedding Space, returning them as a list in order they were executed

        """
        from .featrix_job import FeatrixJob

        results = self.fc.api.op("es_get_training_jobs", embedding_space_id=str(self.id))
        return ApiInfo.reclass(FeatrixJob, results, fc=self.fc)

    def explorer_training_jobs(
            self,
            wait_for_creation: bool = False,
            wait_for_training_job: "FeatrixJob" = None,  # noqa
            max_wait: int = 5
    ) -> List["FeatrixJob"]: # noqa forward ref
        """
        Get training jobs for an explorer project creation

        Args:
            wait_for_creation: If True, wait for the jobs to be created
            wait_for_training_job: If provided, wait for this job to complete first
            max_wait: Maximum number of times to wait for the jobs to be created

        Returns:
            List of FeatrixJob objects
        """
        from .featrix_job import FeatrixJob  # noqa

        if wait_for_training_job:
            # Wait for this job first -- it will be kicking the rest off.
            wait_for_training_job.wait_for_completion(
                f"Waiting for job watcher completion (id {wait_for_training_job.id})...")
        if self.es_neural_attrs is None:
            print("es neurals are none!")
            es = self.by_id(str(self.id), fc=self.fc)
        else:
            es = self
        if es.es_neural_attrs is None:
            raise RuntimeError(f"EmbeddingSpace did not have neural attributes after training ({es.id})")
        jobs = es.find_training_jobs()
        if wait_for_creation:
            cnt = 0
            columns = len(es.es_neural_attrs.get('col_order', []))
            while len(jobs) != columns:
                if cnt == max_wait:
                    raise RuntimeError(f"Model Jobs not scheduled correctly: {len(jobs)} of {columns} found")
                cnt += 1
                display_message("Waiting for model trailing jobs to all be scheduled... "
                                f"({len(jobs)} / {columns})")
                time.sleep(5)
                jobs = es.find_training_jobs()
        return jobs

    def models(self, force: bool = False) -> List[FeatrixModel]:
        """
        Retrieve all models for this embedding space.  If we don't have them cached, or if force is True, we'll
        get them from a server call synchronously.
        """

        since = None
        if self._models_cache_updated is not None and not force:
            since = self._models_cache_updated
            result = self.fc.check_updates(model=self._models_cache_updated)
            if result.model is False:
                return list(self._models_cache.values())
        self._models_cache_updated = datetime.utcnow()
        results = self.fc.api.op(
            "es_get_models", embedding_space_id=str(self._id), since=since
        )
        models = ApiInfo.reclass(FeatrixModel, results, fc=self.fc)
        for model in models:
            # FIXME: The model isn't getting serialized correctly
            if hasattr(model, "_id"):
                model.id = getattr(model, "_id")
            self._models_cache[str(model.id)] = model
        return models

    def model(
        self, model_id: str | PydanticObjectId, force: bool = False
    ) -> FeatrixModel:
        """
        Get a model by it's id from the cache or server (if not in the cache or force is True)

        Arguments:
            model_id: The ID of the model to retrieve
            force: If True, force a refresh of the model from the server

        Returns:
            FeatrixModel object
        """
        model_id = str(model_id)
        if not force:
            if model_id in self._models_cache:
                return self._models_cache[model_id]
        result = self.fc.api.op(
            "es_get_model", embedding_space_id=str(self.id), model_id=str(model_id)
        )
        model = ApiInfo.reclass(FeatrixModel, result, fc=self.fc)
        self._models_cache[model_id] = model
        return model

    def histogram(self) -> EmbeddingDistanceResponse:
        """
        Make call to get the histogram of this embedding space from the server.
        """
        results = self.fc.api.op("es_get_histogram", embedding_space_id=str(self.id))
        return results

    def distance(self) -> EmbeddingDistanceResponse:
        """
        Make the call to get the distance of the embedding space from the server.
        """
        results = self.fc.api.op("es_get_distance", embedding_space_id=str(self.id))
        return results

    def delete(self) -> "FeatrixEmbeddingSpace":
        """
        Delete this embedding space off the server
        """
        result = self.fc.api.op("es_delete", embedding_space_id=str(self.id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, result, fc=self.fc)
