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

import logging
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from featrixclient.featrix_job import FeatrixJob
from pydantic import Field
from pydantic import PrivateAttr

from .api_urls import ApiInfo
from .exceptions import FeatrixException, FeatrixJobFailure
from .featrix_embedding_space import FeatrixEmbeddingSpace
from .featrix_upload import FeatrixUpload
from .models import Project, NewExplorerArgs, ESCreateArgs
from .models import ProjectType
from .models import PydanticObjectId
from .models.project import AllFieldsResponse, ProjectDeleteResponse
from .utils import display_message
from .config import settings

logger = logging.getLogger(__name__)


class FeatrixProject(Project):
    """
    This class represents a project in Featrix. A project provides organization for a users embedding spaces and
    neural functions.  It allows the user to set default settings for the embedding space, as well as associate
    data files to be used in training the embedding space.


    Generally you can just use the project references you get
    back from various methods in the Featrix main class (like create_project) but
    all the operations that are performed on a project at the Featrix level are supported in this class by
    individual methods.

    """
    _fc: Optional[Any] = PrivateAttr(default=None)
    """Reference to the Featrix class  that retrieved or created this project, used for API calls/credentials"""

    # We keep the jobs at the project level -- even though some jobs are in embeddings, some in uploads, etc.
    _jobs_cache: Dict[str, FeatrixJob] = PrivateAttr(default_factory=dict)
    _jobs_cache_updated: Optional[datetime] = PrivateAttr(default=None)
    _embedding_spaces_cache: Dict[str, FeatrixEmbeddingSpace] = PrivateAttr(
        default_factory=dict
    )
    _embedding_spaces_cache_updated: Optional[datetime] = PrivateAttr(
        default=None
    )
    _all_fields_cache: List[AllFieldsResponse] = PrivateAttr(
        default_factory=list
    )
    _all_fields_cache_updated: Optional[datetime] = PrivateAttr(default=None)

    @classmethod
    def new(
        cls,
        fc: Any,
        name: Optional[str] = None,
        project_type: ProjectType = ProjectType.SDK,
        user_meta: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Load or create a project to work with.  If the project_id is passed, we look up an existing project,
        otherwise we check if the named project already exists, and if it doesn't we create a new one.

        Arguments:
            fc: Featrix client class
            name: optional name of the project to look up or create by
            project_type: ProjectType defaulting to SDK
            user_meta: optional user meta if standing up a new project (or it will be added)
            tags: optional list of tags to add to the project
        """
        project = fc.api.op(
            "project_create", name=name, type=project_type.name, tags=tags or [], user_meta=user_meta or {}
        )
        return ApiInfo.reclass(cls, project, fc=fc)

    @classmethod
    def all(cls, fc: Any) -> List[FeatrixProject]:
        """
        Retrieve all known projects from the Featrix server.

        Arguments:
            fc:  Featrix class

        Returns:
            List of FeatrixProject instances
        """
        projects = fc.api.op("project_get_all")
        return ApiInfo.reclass(cls, projects, fc=fc)

    @classmethod
    def by_id(cls, project_id, fc) -> FeatrixProject:
        """
        Retrieve a project from the Featrix server by its id (`FeatrixProject.id`)

        Arguments:
            project_id: str - the id of the project to retrieve
            fc: Featrix class

        Returns:
            FeatrixProject instance
        """
        return ApiInfo.reclass(
            cls, fc.api.op("project_get", project_id=project_id), fc=fc
        )

    def ready(self, wait_for_completion: bool = False) -> bool:
        """
        Check to see if all of the data files that are contained in this project are ready to be used for training.

        Arguments:
            wait_for_completion: bool - if True, will wait for the data files to be ready before returning

        Returns:
            bool - True if all data files are ready for training, False otherwise
        """
        print(f"Called project.ready({wait_for_completion}")
        not_ready = []
        if len(self.associated_uploads) == 0:
            project = self.by_id(self.id, self._fc)
            if len(project.associated_uploads) == 0:
                raise FeatrixException(f"Project {self.name} ({self.id}) has no associated uploads/datafiles")
            return project.ready()
        for ua in self.associated_uploads:
            upload = FeatrixUpload.by_id(ua.upload_id, self._fc)
            if upload.ready_for_training is False:
                not_ready.append(upload)
        print(f"Initially the not ready count is {len(not_ready)}")
        if len(not_ready) == 0:
            return True
        elif wait_for_completion is False:
            print("No waiting -- returning false")
            return False
        for up in not_ready:
            while up.ready_for_training is False:
                display_message(f"Waiting for upload {up.filename} to be ready for training")
                time.sleep(5)
                up = up.by_id(up.id, self._fc)
        display_message("Uploads processed, project ready for training")
        return True

    def create_embedding_space(
            self,
            name: Optional[str] = None,
            credit_budget: int = 3,
            wait_for_completion: bool = False,
            encoder: Optional[Dict] = None,
            ignore_cols: Optional[List[str] | str] = None,
            focus_cols: Optional[List[str] | str] = None,
            **kwargs,
    ) -> Tuple["FeatrixEmbeddingSpace", FeatrixJob]:  # noqa forward ref
        """
        Create a new embedding space in the project specified (FeatrixProject or
        id of a project).

        You do not need to clean nulls or make the data numeric; simply pass in strings or missing values.

        If the wait_for_completion flag is set, this will be synchronous and print periodic messages to the console
        as the embedding space is trained.  Note that the jobs are enqueued and running so if the notebook is
        interrupted, reset or crashes, the training will still complete and can be queried by using the methods later.

        In either case this returns a tuple of the `FeatrixEmbeddingSpace` object and the `FeatrixJob` object that
        created or is creating the job.  `FeatrixEmbeddingSpace.training_state` shows the state of the
        embedding space, but the `Job` has detailed information about the current status.

        Arguments:
            project: FeatrixProject or str id of the project to use; if none passed, we create the project
            name: str -- name of embedding space
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
            Tuple(FeatrixEmbeddingSpace, FeatrixJob) -- the featrix model and the jobs associated with training the model
                         if wait_for_completion is True, the model returned will be fully trained, otherwise the
                         caller will need ot check on the progress of the jobs and update the model when they are
                         complete.
        """
        from .featrix_embedding_space import FeatrixEmbeddingSpace

        if self.ready(wait_for_completion=wait_for_completion) is False:
            raise FeatrixException("Project not ready for training, datafiles still being processed")
        es, job = FeatrixEmbeddingSpace.new_embedding_space(
            fc=self._fc,
            project=self,
            name=name,
            credit_budget=credit_budget,
            encoder=encoder,
            ignore_cols=ignore_cols,
            focus_cols=focus_cols,
            **kwargs
        )
        if wait_for_completion:
            job = job.wait_for_completion("Training Embedding Space: ")
        if job.error:
            raise FeatrixJobFailure(job)
        es = FeatrixEmbeddingSpace.by_id(job.embedding_space_id, self)
        return es, job

    def save(self) -> FeatrixProject:
        """
        Save the project to the Featrix server including anything changed (such as meta or the name).

        Returns:
            FeatrixProject instance
        """
        project = self._fc.api.op("project_update", self)
        return ApiInfo.reclass(FeatrixProject, project, fc=self._fc)

    def jobs(self, stale_timeout: int = settings.stale_timeout) -> List[FeatrixJob]:
        """
        Retrieve the jobs associated with this project.  If the jobs have already been retrieved, they will be
        returned from the cache unless force is True.

        Arguments:
            stale_timeout: seconds to allow cache to age before refresh

        Returns:
            List of FeatrixJob instances
        """
        since = None
        if (
                self._jobs_cache_updated is None or
                (datetime.utcnow() - self._jobs_cache_updated).total_seconds() > stale_timeout
        ):
            since = self._jobs_cache_updated or datetime(2024, 1, 1)
            self._jobs_cache_updated = datetime.utcnow()
            results = self._fc.api.op(
                "project_get_jobs", project_id=str(self.id), since=since
            )
            job_list = ApiInfo.reclass(FeatrixJob, results, fc=self._fc)
            for job in job_list:
                self._jobs_cache[str(job.id)] = job
        return list(self._jobs_cache.values())

    def job(self, job_id: str | PydanticObjectId, stale_timeout: int = settings.stale_timeout) -> FeatrixJob:
        """
        Get a job by its Job id, possibly refreshing the cache if force is True.

        Arguments:
            job_id: str - the id of the job to retrieve
            stale_timeout: seconds to allow cache to age before refresh

        Returns:
            FeatrixJob instance
        """
        job_id = str(job_id)
        if (
                job_id not in self._jobs_cache or
                self._jobs_cache_updated is None or
                (datetime.utcnow() - self._jobs_cache_updated).total_seconds() > stale_timeout
        ):
            self.jobs(stale_timeout=-1)
        if job_id in self._jobs_cache:
            return self.job_cache[job_id]
        raise RuntimeError(f"No such job {job_id} in project {self.name} ({self.id})")

    def embedding_spaces(self, stale_timeout: int = settings.stale_timeout) -> List[FeatrixEmbeddingSpace]:
        """
        Retrieve the embedding spaces associated with this project.  If the embedding spaces have already been retrieved,
        they will be returned from the cache unless force is True.

        Arguments:
            stale_timeout: seconds to allow cache to age before refresh

        Returns:
            List of FeatrixEmbeddingSpace instances
        """
        since = None
        if (
                self._embedding_spaces_cache_updated is None or
                (datetime.utcnow() - self._embedding_spaces_cache_updated).total_seconds() > stale_timeout
        ):
            since = self._embedding_spaces_cache_updated or datetime(2024, 1, 1)

            self._embedding_spaces_cache_updated = datetime.utcnow()
            results = self._fc.api.op(
                "project_get_embedding_spaces", project_id=str(self.id), since=since
            )
            es_list = ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=self._fc)
            for es in es_list:
                self._embedding_spaces_cache[str(es.id)] = es
        return list(self._embedding_spaces_cache.values())

    def embedding_space(
            self,
            embedding_space_id: str | PydanticObjectId,
            stale_timeout: int = settings.stale_timeout
    ) -> FeatrixEmbeddingSpace:
        """
        Get an embedding space by its id, possibly refreshing the cache if force is True.

        Arguments:
            embedding_space_id: str - the id of the embedding space to retrieve
            stale_timeout: seconds to allow cache to age before refresh

        Returns:
            FeatrixEmbeddingSpace instance
        """
        embedding_space_id = str(embedding_space_id)
        if (
            embedding_space_id not in self._embedding_spaces_cache or
            self._embedding_spaces_cache_updated is None or
            (datetime.utcnow() - self._embedding_spaces_cache_updated).total_seconds() > stale_timeout
        ):
            self.embedding_spaces(stale_timeout=-1)
        if embedding_space_id in self._embedding_spaces_cache:
            return self._embedding_spaces_cache[embedding_space_id]
        raise RuntimeError(f"No such embedding space {embedding_space_id} in project {self.name} ({self.id})")

    def neural_functions(
            self,
            embedding_space: FeatrixEmbeddingSpace = None,
            stale_timeout: int = settings.stale_timeout
    ):
        """
        This is a convenience function that allows the user to get all the neural_functions for all their
        embeddings directly from the project (possibly pulling data from the server)

        Arguments:
            embedding_space:  Get the models for the referenced embedding space, or if none, all of them
            stale_timeout: number of seconds to allow cache to age before refeshing it

        Returns:
            List of FeatrixNeuralFunction instances across this project's embedding spaces
        """
        if embedding_space:
            if embedding_space.project_id != self.id:
                raise RuntimeError(f"Embedding space {embedding_space.id} belongs to "
                                   f"project {embedding_space.project_id} not this project ({self.name}, id={self.id}")
            embeddings = [embedding_space]
        else:
            embeddings = self.embedding_spaces(stale_timeout=stale_timeout)
        model_list = []
        for es in embeddings:
            if self._fc.debug:
                print(f"Calling es.models for {es.id}")
            model_list += es.neural_functions(stale_timeout=stale_timeout)
        return model_list

    def find_neural_function(self, ident: str, stale_timeout: int = settings.stale_timeout) -> "FeatrixNeuralFunction":
        """
        Find a model by its id across all embedding spaces in this project.  The stale timeout tells us how old to
        allow the cache to be before refreshing it -- -1 can be used to force it always.

        Arguments:
            ident: str - the id of the model to find
            stale_timeout:  number of seconds to allow cache to age before refeshing it
        Returns:
            FeatrixNeuralFunction instance or None if not found
        """
        from .featrix_neural_function import FeatrixNeuralFunction  # noqa forward ref

        if (
                self._jobs_cache_updated is None or
                (datetime.utcnow() - self._jobs_cache_updated).total_seconds() > stale_timeout
        ):
            self.embedding_spaces(stale_timeout=-1)
            self.neural_functions(stale_timeout=-1)

        for es in self._embedding_spaces_cache.values():
            if ident in es._models_cache:
                return es._models_cache[ident]
        raise RuntimeError(f"No such model {ident} in project {self.name} ({self.id})")

    def fields(self, stale_timeout: int = settings.stale_timeout):
        """
        Retrieve all fields that are in data files associated with this project.  If the cache is stale
        by way of stale_timeout (default 1 hour), it is refreshed.
        """
        if (
                self._all_fields_cache_updated is None or
                (datetime.utcnow() - self._all_fields_cache_updated).total_seconds() > stale_timeout
        ):
            self._all_fields_cache_updated = datetime.utcnow()
            results = self._fc.api.op("project_get_fields", project_id=str(self.id))
            self._all_fields_cache = ApiInfo.reclass(AllFieldsResponse, results, fc=self._fc)

        return self._all_fields_cache

    def associate(
        self,
        upload: FeatrixUpload,
        label: Optional[str] = None,
        sample_row_count: int = 0,
        sample_percentage: float = 1.0,
        drop_duplicates: bool = True,
    ):
        """
        Associate a FeatrixUpload with this project.

        Arguments:
            upload: FeatrixUpload - the upload to associate
            label: str - optional label to give the association
            sample_row_count: int - number of rows to sample
            sample_percentage: float - percentage of rows to sample
            drop_duplicates: bool - whether to drop duplicates

        Returns:
            FeatrixProject updated instance
        """
        if sample_row_count is None and sample_percentage is None:
            sample_percentage = 1
        if label is None:
            label = upload.filename
        results = self._fc.api.op(
            "project_associate_file",
            project_id=str(self.id),
            upload_id=str(upload.id),
            label=label,
            sample_row_count=sample_row_count,
            sample_percentage=sample_percentage,
            drop_duplicates=drop_duplicates,
        )
        return ApiInfo.reclass(FeatrixProject, results, fc=self._fc)

    def add_mapping(self, source_label, target_label, *args):
        """
        Add a mapping between fields in the source and target data files associated with this project.

        Arguments:
            source_label: str - the label of the source data file
            target_label: str - the label of the target data file
            *args: tuple - each tuple should be a pair of fields to map

        Returns:
            FeatrixProject updated instance
        """
        """
        FIXME: interface?  Right now we are just pulling in args where we expect each as a tuple of
        (source_field, target_field)
        """
        mappings = dict(target=target_label, source=source_label, fields=[])
        for s in args:
            mappings["fields"].append({"source_field": s[0], "target_field": s[1]})
        results = self._fc.api.op(
            "project_add_mapping", project_id=str(self.id), mappings=mappings
        )
        return ApiInfo.reclass(FeatrixProject, results, fc=self._fc)

    def add_ignore_columns(self, columns: List[str] | str, *args):
        """
        Add columns to ignore when training the embedding space.

        Arguments:
            columns: List[str] | str - the columns to ignore or a column name
            *args:  additional column names -- allows calling .add_ignore_columns("col1", "col2", "col3") or with a list

        Returns:
            FeatrixProject updated instance
        """
        if isinstance(columns, str):
            columns = [columns]
            for a in args:
                columns.append(str(a))
        results = self._fc.api.op(
            "project_add_ignore_columns",
            project_id=str(self.id),
            columns=columns,
        )
        return ApiInfo.reclass(FeatrixProject, results, fc=self._fc)

    def delete(self):
        """
        Delete the project from the Featrix server.  This will remove all associated data files, embedding spaces,

        Returns:
            ProjectDeleteResponse
        """
        result = self._fc.api.op("project_delete", project_id=str(self.id))
        self._jobs_cache = dict()
        self._embedding_spaces_cache = dict()
        self._all_fields_cache = []
        self._fc.drop_project(self.id)
        return result

    def new_explorer(self, name: str,  training_credits_budgeted: float = 50, **kwargs):
        es_create_args = FeatrixEmbeddingSpace.create_args(
            str(self.id),
            name,
            training_budget_credits=training_credits_budgeted,
            **kwargs
        )
        """
        Create a new explorer function in this project.  This will create a new embedding space and a new neural function
        for each field. This is done with a series of jobs automatically, see the Featrix.new_explorer method for more
        information.  
        """
        dispatches = self._fc.api.op(
            "job_chained_new_explorer",
            name=name,
            project_id=str(self.id),
            training_credits_budgeted=training_credits_budgeted,
            embedding_space_create=es_create_args
        )
        jobs = [FeatrixJob.from_job_dispatch(dispatch, self._fc) for dispatch in dispatches]
        es = FeatrixEmbeddingSpace.by_id(str(jobs[0].embedding_space_id), self._fc)
        return es, jobs
