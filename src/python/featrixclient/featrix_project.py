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
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from featrixclient.featrix_job import FeatrixJob
from pydantic import PrivateAttr

from .api_urls import ApiInfo
from .config import settings
from .exceptions import FeatrixException
from .exceptions import FeatrixJobFailure
from .exceptions import FeatrixNotReadyException
from .featrix_embedding_space import FeatrixEmbeddingSpace
from .featrix_neural_function import FeatrixNeuralFunction
from .featrix_upload import FeatrixUpload
from .models import Project
from .models import ProjectType
from .models import PydanticObjectId
from .models.project import AllFieldsResponse
from .utils import display_message

logger = logging.getLogger(__name__)


class FeatrixProject(Project):
    """
    Represents a project in Featrix, organizing embedding spaces and neural functions. A project allows setting default embedding space settings and associating data files for training.

    You can typically use project references returned by various Featrix methods (e.g., `create_project`). However, this class provides methods for all project-related operations.

    Retrieve a project by ID with `by_id()`, and refresh an existing project reference with `.refresh()` to get the latest version.

    The `.ready()` method checks if the project is ready for model creation/training, indicating if associated data files have been processed. If `wait_for_completion=True`, it will block with status messages until all files are ready.
    """


    _fc: Optional[Any] = PrivateAttr(default=None)
    """Reference to the Featrix class  that retrieved or created this project, used for API calls/credentials"""

    # We keep the jobs at the project level -- even though some jobs are in embeddings, some in uploads, etc.
    _jobs_cache: Dict[str, FeatrixJob] = PrivateAttr(default_factory=dict)
    _jobs_cache_updated: Optional[datetime] = PrivateAttr(default=None)
    _embedding_spaces_cache: Dict[str, FeatrixEmbeddingSpace] = PrivateAttr(
        default_factory=dict
    )
    _embedding_spaces_cache_updated: Optional[datetime] = PrivateAttr(default=None)
    _all_fields_cache: List[AllFieldsResponse] = PrivateAttr(default_factory=list)
    _all_fields_cache_updated: Optional[datetime] = PrivateAttr(default=None)

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
    def new(
        cls,
        fc: Any,
        name: Optional[str] = None,
        project_type: ProjectType = ProjectType.SDK,
        user_meta: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Load or create a project. If `project_id` is provided, an existing project is retrieved. Otherwise, the project is looked up by name, and if it doesn't exist, a new one is created.

        Args:
            fc (FeatrixClient): The Featrix client.
            name (str | None): Optional name of the project to look up or create.
            project_type (ProjectType): The type of project, defaulting to SDK.
            user_meta (dict | None): Optional metadata for a new project.
            tags (list | None): Optional list of tags to add to the project.
        """

        project = fc.api.op(
            "project_create",
            name=name,
            type=project_type.name,
            tags=tags or [],
            user_meta=user_meta or {},
        )
        return ApiInfo.reclass(cls, project, fc=fc)

    @classmethod
    def all(cls, fc: Optional["Featrix"] = None) -> List[FeatrixProject]:  # noqa forward ref
        """
        Retrieve all known projects from the Featrix server.

        Arguments:
            fc:  Featrix class

        Returns:
            List of FeatrixProject instances
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()
        projects = fc.api.op("project_get_all")
        return ApiInfo.reclass(cls, projects, fc=fc)

    @classmethod
    def by_id(cls, project_id, fc: Optional["Featrix"] = None) -> FeatrixProject:  # noqa forward ref
        """
        Retrieve a project from the Featrix server by its id (`FeatrixProject.id`)

        Arguments:
            project_id: str - the id of the project to retrieve
            fc: Featrix class

        Returns:
            FeatrixProject instance
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()

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
        # print(f"Called project.ready({wait_for_completion})")
        not_ready = []
        if len(self.associated_uploads) == 0:
            project = self.by_id(self.id, self._fc)
            if len(project.associated_uploads) == 0:
                return False
                # raise FeatrixException(
                #     f"Project {self.name} ({self.id}) has no associated uploads/datafiles"
                # )
            return project.ready()
        
        for ua in self.associated_uploads:
            upload = FeatrixUpload.by_id(ua.upload_id, self._fc)
            if upload.ready_for_training is False:
                not_ready.append(upload)
        # print(f"Initially the not ready count is {len(not_ready)}")
        if len(not_ready) == 0:
            return True
        elif wait_for_completion is False:
            # print("No waiting -- returning false")
            return False
        for up in not_ready:
            while up.ready_for_training is False:
                display_message(
                    f"Waiting for upload {up.filename} to be ready for training"
                )
                time.sleep(5)
                up = up.by_id(up.id, self._fc)
        display_message("Uploads processed, project ready for training")
        return True

    def create_embedding_space(
        self,
        name: Optional[str] = None,
        wait_for_completion: bool = True,
        encoder: Optional[Dict] = None,
        ignore_cols: Optional[List[str] | str] = None,
        focus_cols: Optional[List[str] | str] = None,
        **kwargs,
    ) -> "FeatrixEmbeddingSpace":  # noqa forward ref
        """
        Create a new embedding space in a specified project.

        Data can include strings and missing values; no need for cleaning. If `wait_for_completion` is set to `True`, the process is synchronous with periodic status updates. The training will complete even if interrupted, and the status can be checked later.

        This returns a tuple with the `FeatrixEmbeddingSpace` object and the `FeatrixJob` object responsible for the training. The `FeatrixEmbeddingSpace.training_state` shows the embedding space's state, while the `Job` provides detailed status information.

        Args:
            project (FeatrixProject | str | None): The project to use; a new project is created if not provided.
            name (str): Name of the embedding space.
            wait_for_completion (bool): Run synchronously with status updates.
            encoder (dict | None): Optional encoder overrides for the embedding space.
            ignore_cols (list | str | None): Columns to ignore in training (list or comma-separated string).
            focus_cols (list | str | None): Columns to focus on in training (list or comma-separated string).
            **kwargs: Additional arguments for `ESCreateArgs`, e.g., `rows=1000`.

        Exceptions:
            FeatrixNotReadyException: Thrown if the project data files are missing or not finished processing.
            
        Returns:
            FeatrixEmbeddingSpace: The embedding space and associated training job.
        """


        from .featrix_embedding_space import FeatrixEmbeddingSpace

        if self.ready(wait_for_completion=wait_for_completion) is False:
           raise FeatrixNotReadyException(
               "Project not ready for creating an embedding space, data files still being processed or not present."
           )
        es = FeatrixEmbeddingSpace.new_embedding_space(
            fc=self._fc,
            project=self,
            name=name,
            encoder=encoder,
            ignore_cols=ignore_cols,
            focus_cols=focus_cols,
            **kwargs,
        )
        if wait_for_completion:
            jobs = es.get_jobs()
            for idx, job in enumerate(jobs):
                job = job.wait_for_completion("Training Embedding Space: ")
                if job.error:
                    raise FeatrixJobFailure(job)
        return es.refresh()

    def save(self) -> FeatrixProject:
        """
        Save the project to the Featrix server including anything changed (such as meta or the name).

        Returns:
            FeatrixProject instance
        """
        project = self._fc.api.op("project_update", self)
        return ApiInfo.reclass(FeatrixProject, project, fc=self._fc)

    def jobs(self) -> List[FeatrixJob]:
        """
        Retrieve the jobs associated with this project.  If the jobs have already been retrieved, they will be
        returned from the cache unless force is True.

        Returns:
            List of FeatrixJob instances
        """
        results = self._fc.api.op("project_get_jobs", project_id=str(self.id))
        job_list = ApiInfo.reclass(FeatrixJob, results, fc=self._fc)
        return job_list

    def job_by_id(
        self,
        job_id: str | PydanticObjectId
    ) -> FeatrixJob:
        """
        Get a job by its Job id, possibly refreshing the cache if force is True.

        Arguments:
            job_id: str - the id of the job to retrieve

        Returns:
            FeatrixJob instance
        """
        job_id = str(job_id)
        job_list = self.jobs()
        for job in job_list:
            if job.id == job_id:
                return job
        raise RuntimeError(f"No such job {job_id} in project {self.name} ({self.id})")

    def embedding_spaces(self) -> List[FeatrixEmbeddingSpace]:
        """
        Retrieve the embedding spaces associated with this project.  If the embedding spaces have already been retrieved,
        they will be returned from the cache unless force is True.

        Returns:
            List of FeatrixEmbeddingSpace instances
        """
        results = self._fc.api.op("project_get_embedding_spaces", project_id=str(self.id))
        es_list = ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=self._fc)
        return es_list

    def embedding_space_by_id(
        self,
        embedding_space_id: str | PydanticObjectId,
    ) -> FeatrixEmbeddingSpace:
        """
        Get an embedding space by its id, possibly refreshing the cache if force is True.

        Arguments:
            embedding_space_id: str - the id of the embedding space to retrieve

        Returns:
            FeatrixEmbeddingSpace instance
        """
        es_list = self.embedding_spaces()
        for es in es_list:
            if es.id == str(embedding_space_id):
                return es
            
        raise RuntimeError(
            f"No such embedding space {embedding_space_id} in project {self.name} ({self.id})"
        )

    def neural_functions(
        self,
        embedding_space: FeatrixEmbeddingSpace = None
    ):
        """
        This is a convenience function that allows the user to get all the neural_functions for all their
        embeddings directly from the project (possibly pulling data from the server)

        Arguments:
            embedding_space:  Get the models for the referenced embedding space, or if none, all of them

        Returns:
            List of FeatrixNeuralFunction instances across this project's embedding spaces
        """
        if embedding_space:
            if str(embedding_space.project_id) != str(self.id):
                raise RuntimeError(
                    f"Embedding space {embedding_space.id} belongs to "
                    f"project {embedding_space.project_id} not this project ({self.name}, id={self.id}"
                )
            embeddings = [embedding_space]
        else:
            embeddings = self.embedding_spaces()
        model_list = []
        for es in embeddings:
            if self._fc.debug:
                print(f"Calling es.models for {es.id}")
            model_list += es.neural_functions()
        return model_list

    def neural_function_by_id(
        self, 
        ident: str
    ) -> FeatrixNeuralFunction:
        """
        Find a model by its id across all embedding spaces in this project.  The stale timeout tells us how old to
        allow the cache to be before refreshing it -- -1 can be used to force it always.

        Arguments:
            ident: str - the id of the model to find
        Returns:
            FeatrixNeuralFunction instance or None if not found
        """
        assert ident is not None
        assert len(str(ident)) > 0

        nf_list = self.neural_functions()
        for nf in nf_list:
            if nf.id == str(ident):
                return nf

        raise RuntimeError(f"No such model {ident} in project {self.name} ({self.id})")

    def fields(self):
        """
        Retrieve all fields that are in data files associated with this project.
        """
        results = self._fc.api.op("project_get_fields", project_id=str(self.id))
        all_fields = ApiInfo.reclass(
            AllFieldsResponse, results, fc=self._fc
        )

        return all_fields

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
        Delete the project from the Featrix server.  This will remove all associated data files, embedding spaces, neural functions, etc. Proceed with extreme caution.

        Returns:
            ProjectDeleteResponse
        """
        result = self._fc.api.op("project_delete", project_id=str(self.id))
        self._jobs_cache = dict()
        self._embedding_spaces_cache = dict()
        self._all_fields_cache = []
        # self._fc.drop_project(self.id)
        return result
