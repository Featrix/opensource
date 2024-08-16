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
    _models_cache: Dict[str, FeatrixNeuralFunction] = PrivateAttr(default_factory=dict)
    _models_cache_updated: Optional[datetime] = PrivateAttr(default=None)
    _explorer_data: Dict = PrivateAttr(default=None)

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
        credit_budget: int = 3,
        wait_for_completion: bool = False,
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
            training_budget_credits=credit_budget,
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
                job.wait_for_completion(f"Job {job.job_type} (id={job.id}): ")
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

    def get_explorer_data(self, force: bool = False) -> Dict:
        """
        Retrieve the explorer data for this embedding space, which includes the neural attributes

        Args:
            force: If True, force a refresh of the explorer data

        Returns:
            Dict of the explorer data
        """
        if self._explorer_data is None or force:
            self._explorer_data = self._fc.api.op(
                "es_get_explorer", embedding_space_id=str(self.id)
            )
        return self._explorer_data

    def find_training_jobs(self) -> List["FeatrixJob"]:  # noqa forward ref
        """
        Find any/all training jobs for this Embedding Space, returning them as a list in order they were executed

        """
        from .featrix_job import FeatrixJob

        results = self._fc.api.op(
            "es_get_training_jobs", embedding_space_id=str(self.id)
        )
        return ApiInfo.reclass(FeatrixJob, results, fc=self._fc)

    def explorer_training_jobs(
        self,
        wait_for_creation: bool = False,
        wait_for_training_job: "FeatrixJob" = None,  # noqa
        max_wait: int = 5,
    ) -> List["FeatrixJob"]:  # noqa forward ref
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
                f"Waiting for job watcher completion (id {wait_for_training_job.id})..."
            )
        if self.es_neural_attrs is None:
            print("es neurals are none!")
            es = self.by_id(str(self.id), fc=self._fc)
        else:
            es = self
        if es.es_neural_attrs is None:
            raise RuntimeError(
                f"EmbeddingSpace did not have neural attributes after training ({es.id})"
            )
        jobs = es.find_training_jobs()
        if wait_for_creation:
            cnt = 0
            columns = len(es.es_neural_attrs.get("col_order", []))
            while len(jobs) != columns:
                if cnt == max_wait:
                    raise RuntimeError(
                        f"Model Jobs not scheduled correctly: {len(jobs)} of {columns} found"
                    )
                cnt += 1
                display_message(
                    "Waiting for model trailing jobs to all be scheduled... "
                    f"({len(jobs)} / {columns})"
                )
                time.sleep(5)
                jobs = es.find_training_jobs()
        return jobs

    def neural_functions(
        self, stale_timeout: int = settings.stale_timeout
    ) -> List[FeatrixNeuralFunction]:
        """
        Retrieve all models for this embedding space.  If we don't have them cached, or if force is True, we'll
        get them from a server call synchronously.
        """

        since = None
        if (
            self._models_cache_updated is None
            or (datetime.utcnow() - self._models_cache_updated).total_seconds()
            > stale_timeout
        ):
            since = self._models_cache_updated
            self._models_cache_updated = datetime.utcnow()
            results = self._fc.api.op(
                "es_get_models", embedding_space_id=str(self._id), since=since
            )
            models = ApiInfo.reclass(FeatrixNeuralFunction, results, fc=self._fc)
            for model in models:
                # FIXME: The model isn't getting serialized correctly
                if hasattr(model, "_id"):
                    model.id = getattr(model, "_id")
                self._models_cache[str(model.id)] = model
        return list(self._models_cache.values())

    def neural_function(
        self,
        model_id: str | PydanticObjectId,
        stale_timeout: int = settings.stale_timeout,
    ) -> FeatrixNeuralFunction:
        """
        Get a model by it's id from the cache or server (if not in the cache or force is True)

        Arguments:
            model_id: The ID of the model to retrieve
            stale_timeout: The number of seconds to wait before refreshing the cache

        Returns:
            FeatrixNeuralFunction object
        """
        model_id = str(model_id)
        if (
            self._models_cache_updated is None
            or model_id not in self._models_cache
            or (datetime.utcnow() - self._models_cache_updated).total_seconds()
            > stale_timeout
        ):
            result = self._fc.api.op(
                "es_get_model", embedding_space_id=str(self.id), model_id=str(model_id)
            )
            model = ApiInfo.reclass(FeatrixNeuralFunction, result, fc=self._fc)
            self._models_cache[model_id] = model
        if model_id in self._models_cache:
            return self._models_cache[model_id]
        raise RuntimeError(
            f"Model {model_id} not found in Embedding space {self.name} (id={self.id})"
        )

    def create_neural_function(
        self,
        target_fields: str | List[str],
        credit_budget: int = 3,
        wait_for_completion: bool = False,
        encoder: Optional[Dict] = None,
        ignore_cols: Optional[List[str] | str] = None,
        focus_cols: Optional[List[str] | str] = None,
        **kwargs,
    ) -> Tuple[FeatrixNeuralFunction, "FeatrixJob", "FeatrixJob"]:  # noqa forward ref
        """
        Create a new neural function in the given project.  If a project is passed in (can be either a FeatrixProject
        or the id of a project), we use that.  If the project is a string and not an id, we will assume it's a name
        and create a new project (if it's none, we will create a name for the project using target_fields).

        If an embedding space is already trained or being trained in the project, we will use that embedding space
        to train the neural function model, otherwise we will first train an embedding space on the data files included
        in the project.  If a list of datasets are passed into this function, we will first upload and associate
        those files with the project being used.

        If the wait_for_completion flag is set, this will be synchronous and
        print periodic messages to the console or notebook cell.  Note that the jobs are enqueued and running
        so if the notebook is interrupted, reset or crashes, the training will still complete and can be queried
        by using the methods get_neural_function or neural_functions.

        In either case, a tuple is returned that includes the model and two FeatrixJob objects -- the first is
        the embedding space training job, and the second is the model training job.  If the embedding space was
        already training, the first job will be the last training job for that embedding space.

        The caller, in the case where they do not wait for completion, can follow the progress via the jobs objects

        .. code-block:: python

           model, es_training_job, nf_training_job = create_neural_function("field_name")
           if nf_training_job.completed is False:
               nf_training_job = nf_training_job.check()
               print(nf_training_job.incremental_status)


        They can also just wait on the neural function model's field training_state to be set to
        TrainingState.COMPLETED ("trained")

        Arguments:
            target_fields: the field name(s) to target in the prediction
            project: FeatrixProject or str id of the project to use or the name for a new project
            credit_budget(int): the default credit budget for the training
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
            Tuple(FeatrixNeuralFunction, Job, Job) -- the featrix model and the jobs associated with training the model
                         if wait_for_completion is True, the model returned will be fully trained, otherwise the
                         caller will need ot check on the progress of the jobs and update the model when they are
                         complete.
        """
        from .featrix_job import FeatrixJob  # noqa forward ref

        project = self._fc.get_project_by_id(self.project_id)
        if project.ready(wait_for_completion=wait_for_completion) is False:
            raise FeatrixException(
                "Project not ready for training, datafiles still being processed"
            )
        project = project.refresh()

        nf = FeatrixNeuralFunction.new_neural_function(
            fc=self._fc,
            project=project,
            target_field=target_fields,
            credit_budget=credit_budget,
            encoder=encoder,
            ignore_cols=ignore_cols,
            focus_cols=focus_cols,
            embedding_space=self,
            **kwargs,
        )
        if wait_for_completion:
            training_jobs = nf.get_jobs()
            training_jobs[0].wait_for_completion(
                f"Training Neural Function {nf.name}: "
            )
            # If we are leveraging an embedding space we created previously, the job will be marked as finished already
        return nf.refresh()

    def histogram(self) -> EmbeddingDistanceResponse:
        """
        Make call to get the histogram of this embedding space from the server.
        """
        results = self._fc.api.op("es_get_histogram", embedding_space_id=str(self.id))
        return results

    def distance(self) -> EmbeddingDistanceResponse:
        """
        Make the call to get the distance of the embedding space from the server.
        """
        results = self._fc.api.op("es_get_distance", embedding_space_id=str(self.id))
        return results

    def delete(self) -> "FeatrixEmbeddingSpace":
        """
        Delete this embedding space off the server
        """
        result = self._fc.api.op("es_delete", embedding_space_id=str(self.id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, result, fc=self._fc)

    def encode_record(self, upload: pd.DataFrame | str | Path) -> List[Dict]:
        return self.embed_record(upload)

    def embed_record(self, upload: pd.DataFrame | str | Path) -> List[Dict]:
        """
        Use this trained embedding space to create embeddings for rows of data.
        The rows may be in or out of the training set.
        You can use the resulting vector embeddings to train your own models
        outside of Featrix, to cluster data items with the relationships that
        created the embedding transformation on the training data--this is often
        extremely powerful!

        You can also use these embeddings to chain together composite Featrix
        neural functions!
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
