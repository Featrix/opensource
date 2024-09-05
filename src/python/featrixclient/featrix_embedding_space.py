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
#     https://bits.featrix.com/slack
#
#  We'd love to hear from you: bugs, features, questions -- send them along!
#
#     hello@featrix.ai
#
#############################################################################
#
from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from pydantic import PrivateAttr

from .api_urls import ApiInfo
from .config import settings
from .exceptions import FeatrixException
from .featrix_neural_function import FeatrixNeuralFunction
from .models import EmbeddingDistanceResponse
from .models import EmbeddingSpace
from .models import ESCreateArgs
from .models import JobType
from .models import PydanticObjectId
from .models import TrainingState
from .utils import display_message
from .utils import featrix_wrap_pd_read_csv


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

    _fc: Optional[Any] = PrivateAttr(default=None)
    """Reference to the Featrix class that retrieved or created this project, used for API calls/credentials"""

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

    def ready(self):
        return self.training_state == TrainingState.COMPLETED

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

    def get_jobs(
        self, active: bool = True, training: bool = True
    ) -> List["FeatrixJob"]:  # noqa forward ref
        """
        Return a list of jobs that are associated with this model.  By default it will only
        return active (not finished) training jobs, but the caller can use the two arguments to control this.

        Arguments:
            active: bool: If True, only return active jobs
            training: bool: If True, only return training jobs

        Returns:
            List[FeatrixJob]: The list of jobs associated with this model
        """
        from .featrix_job import FeatrixJob  # noqa forward ref

        jobs = []
        for job in FeatrixJob.by_embedding_space(self):
            if active or training:
                if active and job.finished:
                    continue
                if training and job.job_type not in [
                    JobType.JOB_TYPE_ES_CREATE,
                    JobType.JOB_TYPE_ES_TRAIN_MORE,
                ]:
                    continue
            jobs.append(job)
        return jobs

    @classmethod
    def new_embedding_space(
        cls,
        fc: Any,
        project: "FeatrixProject" | str,  # noqa forward ref
        name: Optional[str] = None,
        wait_for_completion: bool = True,
        encoding: Optional[Dict] = None,
        focus_cols: Optional[List[str] | str] = None,
        ignore_cols: Optional[List[str] | str] = None,
        **kwargs,
    ) -> "FeatrixEmbeddingSpace":
        """
        This creates a chained-job to do training first on an embedding space, and then on the predictive model
        within that embedding space.  It returns a tuple which is the two jobs (the first job for the embedding space
        training and the second for the predictive model training).
        """
        from .featrix_job import FeatrixJob
        from .featrix_project import FeatrixProject  # noqa forward ref

        if name is None:
            name = (
                f"{project.name}-{uuid.uuid4()}"
                if isinstance(project, FeatrixProject)
                else f"Project {uuid.uuid4()}"
            )

        es_create_args = cls.create_args(
            str(project.id) if isinstance(project, FeatrixProject) else project,
            name,
            encoding=encoding or {},
            focus_cols=focus_cols or [],
            ignore_cols=ignore_cols or [],
            **kwargs,
        )
        dispatches = fc.api.op("job_es_create", es_create_args)
        jobs = [FeatrixJob.from_job_dispatch(dispatch, fc) for dispatch in dispatches]
        es = cls.by_id(jobs[-1].embedding_space_id, fc)
        if wait_for_completion:
            for job in jobs:
                job.wait_for_completion(f"Job {job.job_type} (job id={job.id}): ")
                if job.error:
                    raise FeatrixException(
                        f"Failed to train embedding space {job.embedding_space_id}: {job.error_msg}"
                    )
        return es.refresh()

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


    def training_jobs(self) -> List["FeatrixJob"]:  # noqa forward ref
        """
        Fetch all training jobs for this Embedding Space, returning them as a list in order they were executed

        """
        from .featrix_job import FeatrixJob

        results = self._fc.api.op(
            "es_get_training_jobs", embedding_space_id=str(self.id)
        )
        return ApiInfo.reclass(FeatrixJob, results, fc=self._fc)


    def neural_functions(self, lambda_filter=None) -> List[FeatrixNeuralFunction]:
        """
        Retrieve all neural functions for this embedding space.
        """
        results = self._fc.api.op(
            "es_get_models", 
            embedding_space_id=str(self._id)
        )
        models = ApiInfo.reclass(FeatrixNeuralFunction, results, fc=self._fc)
        for model in models:
            # FIXME: The model isn't getting serialized correctly
            if hasattr(model, "_id"):
                model.id = getattr(model, "_id")
        if lambda_filter is not None:
            result_list = []
            for model in models:
                if lambda_filter(model):
                    result_list.append(model)
            models = result_list
        return models #list(self._models_cache.values())


    def neural_function_by_id(
        self,
        model_id: str | PydanticObjectId
    ) -> FeatrixNeuralFunction:
        """
        Get a neural function by its id.

        Arguments:
            model_id: The ID of the model to retrieve

        Returns:
            FeatrixNeuralFunction object
        """
        model_id = str(model_id)
        result = self._fc.api.op(
            "es_get_model", embedding_space_id=str(self.id), model_id=str(model_id)
        )
        model = ApiInfo.reclass(FeatrixNeuralFunction, result, fc=self._fc)
        return model

    def create_neural_function(
        self,
        target_field: str,
        target_field_type: Optional[str] = 'auto', # auto | set | scalar
        wait_for_completion: bool = True,
        encoder: Optional[Dict] = None,
        **kwargs,
    ) -> FeatrixNeuralFunction:  # noqa forward ref
        """
        Creates a new neural function -- a predictive model.

        If the wait_for_completion flag is set, this will be synchronous and
        print periodic messages to the console or notebook cell.  Note that the jobs are enqueued and running
        so if the caller is interrupted, reset or crashes, the training will still complete.

        The caller, in the case where they do not wait for completion, can follow the progress via the jobs objects.

        .. code-block:: python

           nf_training_job = create_neural_function("field_name")
           if nf_training_job.completed is False:
               nf_training_job = nf_training_job.check()
               print(nf_training_job.incremental_status)

        Arguments:
            target_fields: the field name(s) to target in the prediction
            files: a list of dataframes or paths to files to upload and associate with the project
                        (optional - if you already associated files with the project, this is redundant)
            wait_for_completion(bool): make this synchronous, printing out status messages while waiting for the
                                    training to complete
            encoder: Optional dictionary of encoder overrides to use for the embedding space
            ignore_cols: Optional list of columns to ignore in the training  (a string of comma separated
                                                                            column names or a list of strings)
            focus_cols: Optional list of columns to focus on in the training (a string of comma separated
                                                                            column names or a list of strings)
            **kwargs -- any other fields to ESCreateArgs() such -- can be called as to specify rows for instance):
                              create_embedding_space(project, name, credits, files, wait_for_completion, rows=1000)

        Returns:
            FeatrixNeuralFunction - the Featrix neural function.
        """
        from .featrix_job import FeatrixJob  # noqa forward ref

        project = self._fc.get_project_by_id(self.project_id)
        if project.ready(wait_for_completion=wait_for_completion) is False:
            raise FeatrixException(
                "Project {self.project_id} not ready for training, datafiles still being processed"
            )
        project = project.refresh()

        nf = FeatrixNeuralFunction.new_neural_function(
            fc=self._fc,
            target_field=target_field,
            target_field_type=target_field_type,
            encoder=encoder,
            embedding_space=self,
            project=project,
            wait_for_completion=wait_for_completion,
            **kwargs,
        )

        return nf


    def delete(self) -> "FeatrixEmbeddingSpace":
        """
        Delete this embedding space off the server
        """
        result = self._fc.api.op("es_delete", embedding_space_id=str(self.id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, result, fc=self._fc)

    def embed_record(self, upload: pd.DataFrame | str | Path) -> List[Dict]:
        """
        Use this trained embedding space to create embeddings for rows of data.
        The rows may be in or out of the training set.
        You can use the resulting vector embeddings to train your own models
        outside of Featrix, to cluster data items with the relationships that
        created the embedding transformation on the training data--this is often
        extremely powerful.

        You can also use these embeddings to chain together composite Featrix
        neural functions.
        """
        from featrixclient.models.job_requests import EncodeRecordsArgs

        if isinstance(upload, pd.DataFrame):
            records = upload.to_dict(orient="records")
        else:
            path = Path(upload)
            if not path.exists():
                raise FeatrixException(f"File {path} does not exist")
            df = featrix_wrap_pd_read_csv(path)
            records = df.to_dict(orient="records")
        encode_args = EncodeRecordsArgs(
            project_id=str(self.project_id),
            embedding_space_id=str(self.id),
            # upload_id=str(upload.id),
            records=records,
        )
        result = self._fc.api.op("job_fast_encode_records", encode_args)
        return result
