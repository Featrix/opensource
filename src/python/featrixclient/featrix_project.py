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

import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from featrixclient.featrix_job import FeatrixJob
from pydantic import Field

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .featrix_embedding_space import FeatrixEmbeddingSpace
from .featrix_upload import FeatrixUpload
from .models import Project
from .models import PydanticObjectId
from .models.project import AllFieldsResponse, ProjectDeleteResponse
from .utils import display_message


class FeatrixProject(Project):
    fc: Optional[Any] = Field(default=None, exclude=True)
    # We keep the jobs at the project level -- even though some jobs are in embeddings, some in uploads, etc.
    jobs_cache: Dict[str, FeatrixJob] = Field(default_factory=dict, exclude=True)
    jobs_cache_updated: Optional[datetime] = Field(default=None, exclude=True)
    embedding_spaces_cache: Dict[str, FeatrixEmbeddingSpace] = Field(
        default_factory=dict, exclude=True
    )
    embedding_spaces_cache_updated: Optional[datetime] = Field(
        default=None, exclude=True
    )
    all_fields_cache: List[AllFieldsResponse] = Field(
        default_factory=list, exclude=True
    )
    all_fields_cache_updated: Optional[datetime] = Field(default=None, exclude=True)

    @classmethod
    def new(
        cls,
        fc: Any,
        name: Optional[str] = None,
        user_meta: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Load or create a project to work with.  If the project_id is passed, we look up an existing project,
        otherwise we check if the named project already exists, and if it doesn't we create a new one.

        Arguments:
            fc: Featrix client class
            name: optional name of the project to look up or create by
            user_meta: optional user meta if standing up a new project (or it will be added)
            tags: optional list of tags to add to the project
        """
        project = fc.api.op(
            "project_create", name=name, tags=tags or [], user_meta=user_meta or {}
        )
        return ApiInfo.reclass(cls, project, fc=fc)

    @classmethod
    def all(cls, fc: Any):
        projects = fc.api.op("project_get_all")
        return ApiInfo.reclass(cls, projects, fc=fc)

    @classmethod
    def by_id(cls, project_id, fc):
        return ApiInfo.reclass(
            cls, fc.api.op("project_get", project_id=project_id), fc=fc
        )

    def ready(self, wait_for_completion: bool = False) -> bool:
        not_ready = []
        if len(self.associated_uploads) == 0:
            project = self.by_id(self.id, self.fc)
            if len(project.associated_uploads) == 0:
                raise FeatrixException(f"Project {self.name} ({self.id}) has no associated uploads/datafiles")
            return project.ready()
        for ua in self.associated_uploads:
            upload = FeatrixUpload.by_id(ua.upload_id, self.fc)
            if upload.ready_for_training is False:
                not_ready.append(upload)
        if len(not_ready) == 0:
            return True
        elif wait_for_completion is False:
            return False
        for up in not_ready:
            while up.ready_for_training is False:
                display_message(f"Waiting for upload {up.filename} to be ready for training")
                time.sleep(5)
                up = up.by_id(up.id, self.fc)
        display_message("Uploads processed, project ready for training")
        return True

    def save(self):
        project = self.fc.api.op("project_update", self)
        return ApiInfo.reclass(FeatrixProject, project, fc=self.fc)

    def jobs(self, force: bool = False):
        since = None
        if self.jobs_cache_updated is not None and not force:
            since = self.jobs_cache_updated
            result = self.fc.check_updates(job_meta=self.jobs_cache_updated)
            if result.job_meta is False:
                return list(self.jobs_cache.values())
        self.jobs_cache_updated = datetime.utcnow()
        results = self.fc.api.op(
            "project_get_jobs", project_id=str(self.id), since=since
        )
        job_list = ApiInfo.reclass(FeatrixJob, results, fc=self.fc)
        for job in job_list:
            self.jobs_cache[str(job.id)] = job
        return job_list

    def job(self, job_id: str | PydanticObjectId, force: bool = False):
        job_id = str(job_id)
        if force or job_id not in self.jobs_cache:
            self.jobs(force=True)
        if job_id in self.jobs_cache:
            return self.job_cache[job_id]

    def embedding_spaces(self, force: bool = False):
        since = None
        if self.embedding_spaces_cache_updated is not None and not force:
            since = self.embedding_spaces_cache_updated
            result = self.fc.check_updates(
                embedding_space=self.embedding_spaces_cache_updated
            )
            if result.embedding_space is False:
                return list(self.embedding_spaces_cache.values())
        self.embedding_spaces_cache_updated = datetime.utcnow()
        results = self.fc.api.op(
            "project_get_embedding_spaces", project_id=str(self.id), since=since
        )
        es_list = ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=self.fc)
        for es in es_list:
            self.embedding_spaces_cache[str(es.id)] = es
        return es_list

    def models(self):
        model_list = []
        for es in self.embedding_spaces():
            if self.fc.debug:
                print(f"Calling es.models for {es.id}")
            model_list += es.models()
        return model_list

    def find_model(self, ident: str):
        for es in self.embedding_spaces():
            if ident in es.models_cache:
                return es.models_cache[ident]
        return None

    def embedding_space(
        self, embedding_space_id: str | PydanticObjectId, force: bool = False
    ):
        embedding_space_id = str(embedding_space_id)
        if force or embedding_space_id not in self.embedding_spaces_cache:
            self.embedding_spaces(force=True)
        if embedding_space_id in self.embedding_spaces_cache:
            return self.embedding_spaces_cache[embedding_space_id]

    def fields(self, force: bool = False):
        # FIXME: should be using the since arg for refresh -- need to add that -- or look at if there have been
        # any new associations added?
        self.all_fields_cache_updated = datetime.utcnow()
        results = self.fc.api.op("project_get_fields", project_id=str(self.id))
        self.all_fields_cache = ApiInfo.reclass(AllFieldsResponse, results, fc=self.fc)
        return self.all_fields_cache

    def associate(
        self,
        upload: FeatrixUpload,
        label: Optional[str] = None,
        sample_row_count: int = 0,
        sample_percentage: float = 1.0,
        drop_duplicates: bool = True,
    ):
        if sample_row_count is None and sample_percentage is None:
            sample_percentage = 1
        if label is None:
            label = upload.filename
        results = self.fc.api.op(
            "project_associate_file",
            project_id=str(self.id),
            upload_id=str(upload.id),
            label=label,
            sample_row_count=sample_row_count,
            sample_percentage=sample_percentage,
            drop_duplicates=drop_duplicates,
        )
        return ApiInfo.reclass(FeatrixProject, results, fc=self.fc)

    def add_mapping(self, source_label, target_label, *args):
        """
        FIXME: interface?  Right now we are just pulling in args where we expect each as a tuple of
        (source_field, target_field)
        """
        mappings = dict(target=target_label, source=source_label, fields=[])
        for s in args:
            mappings["fields"].append({"source_field": s[0], "target_field": s[1]})
        results = self.fc.api.op(
            "project_add_mapping", project_id=str(self.id), mappings=mappings
        )
        return ApiInfo.reclass(FeatrixProject, results, fc=self.fc)

    def add_ignore_columns(self, *columns):
        results = self.fc.api.op(
            "project_add_ignore_columns",
            project_id=str(self.id),
            columns=[str(col) for col in columns],
        )
        return ApiInfo.reclass(FeatrixProject, results, fc=self.fc)

    def delete(self):
        result = self.fc.api.op("project_delete", project_id=str(self.id))
        self.jobs_cache = dict()
        self.embedding_spaces_cache = dict()
        self.all_fields_cache = []
        self.fc.drop_project(self.id)
        return result
