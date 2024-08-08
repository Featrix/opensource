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

__author__ = "Featrix, Inc"

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd

from featrixclient.featrix_embedding_space import FeatrixEmbeddingSpace
from featrixclient.featrix_job import FeatrixJob
from featrixclient.models import TrainingState, ProjectType, JobType, PydanticObjectId
from .api import FeatrixApi
from .exceptions import FeatrixException, FeatrixJobFailure
from .featrix_neural_function import FeatrixNeuralFunction
from .featrix_project import FeatrixProject
from .featrix_upload import FeatrixUpload
from .version import version, publish_time

__version__ = f"{version}: published at {publish_time}"

logger = logging.getLogger("featrix-client")

KEY_FILE_LOCATION = Path("~/.featrix.key").expanduser()

MAX_RETRIES = 10
MAX_TIMEOUT_RETRIES = 20


class Featrix:
    """
    The Featrix class provides access to all the facilities of Featrix.
    """

    debug: bool = False

    def __init__(
        self,
        url: str = "https://app.featrix.com",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        key_file: Path | str = KEY_FILE_LOCATION,
        allow_unencrypted_http: bool = False,
    ):
        """
        Create a Featrix client object.

        The caller has to provide authentication credentials. These are API Keys, generated in the Featrix UI, found at
        https://app.featrix.com.  An API Key will consist of a client id and a client secret.  Both are needed to
        use this interface.  The credentials can be provided to this interface three different ways (only one is
        required!):
        1) The client_id/client_secret arguments
        2) Having the client id and secret set in the environment variables FEATRIX_CLIENT_ID and
        FEATRIX_CLIENT_SECRET, typically put in the users .zshrc/.bashrc or Windows environment.
        3) Having the client id and secret set in the file ${HOME}/.featrix.key (or a file specified by
        the key_file argument below).  The format of this file is the same as the environment, two lines
        that contain:

           FEATRIX_CLIENT_ID=xxxxxxxxxx
           FEATRIX_CLIENT_SECRET=xxxxxxxxxx

        To generate an API Key, register an account at https://app.featirx.com, and once logged in, under
        your Profile menu in the top right corner, select ``Manage API Keys``.

        Args:
            url(str): The url of the Featrix server you are using. The default is https://app.featrix.com
            client_id (str | None): the client id of the API Key to use for authentication
            client_secret (str | None): The client secret of the API Key to use for authentication
            key_file (str | Path | None): Alternate filename of a file containing your API Key credentials
            allow_unencrypted_http (bool): Allow connection to a non-https (unencrypted) address (used for development)

        Returns
        -------
        A Featrix object ready to access/create your neural functions and preform predictive AI queries against
        those models.
        """
        self._projects = {}
        # by name and by id ?
        self._library = {}
        self._uploads = {}
        self._check_debug()
        self.api = FeatrixApi.new(
            self,
            url=url,
            client_id=client_id,
            client_secret=client_secret,
            key_file=key_file,
            allow_unencrypted_http=allow_unencrypted_http,
            debug=self.debug,
        )

    def _store_project(self, project: FeatrixProject):
        self._projects[str(project.id)] = project

    def projects(self) -> List[FeatrixProject]:
        """
        Return all the projects in your account as a list.  Each project has a name and a list of associated
        data sets (associated_uploads) that are part of that project.  If there are mappings that exist
        between multiple data sets, they will also be listed in the mappings field.  Additionally, if there are
        columns in these data sets that are being ignored by this project, they will be in ignore_cols.

        Returns:
            List[FeatrixProject] -- A list of projects.
        """
        projects = FeatrixProject.all(self)
        for project in projects:
            if self.debug:
                print(f"Found project: {project.model_dump_json(indent=4)}")
            self._store_project(project)
        return projects

    def get_project_by_id(self, project_id: str | PydanticObjectId) -> FeatrixProject | None:
        """
        Find a project in the projects cache by its id (FeatrixProject.id)

        Returns:
            FeatrixProject: The project object found in the cache
        """
        result = self._projects.get(str(project_id))
        if result is None:
            self.projects()
            result = self._projects.get(str(project_id))
        return result
     
    def get_project_by_name(self, name: str) -> FeatrixProject | List[FeatrixProject] | None:
        """
        Find a project in the projects cache by its name (FeatrixProject.name)

        Returns:
            FeatrixProject: The project object found in the cache
        """
        self.projects()
        matches = []
        for _, v in self._projects.items():
            if v.name == name:
                matches.append(v)
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            # uh ohh.
            return matches
        return matches[0]

    def create_project(
        self,
        name: Optional[str] = None,
        user_meta: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> FeatrixProject:
        """
        Create a new project and make it the current project using the name provided. It creates the project
        and sets the current project of the Featrix class to that project.

        Arguments:
            name:  Optional name of the project, otherwise it will be auto-named
            user_meta: Optional dictionary of user metadata to associate with the project
            tags: Optional list of tags to associate with the project

        Returns:
            FeatrixProject - the new project object created

        """
        project = FeatrixProject.new(self, name, user_meta=user_meta, tags=tags)
        self._store_project(project)
        return project

    def drop_project(self, project: FeatrixProject | str) -> None:
        """
        Drop a project from the cache of projects.  This will not delete the project from the Featrix system,
        just remove it from the cache of projects in the Featrix object.

        Args:
            project: The project object or the id of the project to drop from the cache
        """
        project_id = str(project.id) if isinstance(project, FeatrixProject) else project
        if project_id in self._projects:
            del self._projects[project_id]

    def get_uploads(self) -> None:
        """
        Get all the FeatrixUpload entries that describe files the user has uploaded to the Featrix system.
        These are stored in the Featrix cache, and can be retrieved by the `get_upload` method.
        """
        uploads = FeatrixUpload.all(self)
        for upload in uploads:
            self._library[upload.filename] = upload
            self._uploads[str(upload.id)] = upload

    def get_upload(self, upload_id: str = None, filename: str = None, reload: bool = True) -> FeatrixUpload:
        """
        Return the FeatrixUpload object (FeatrixUpload) for the given upload id or filename.

        Args:
            upload_id (str): Upload id of an upload to find in the library (FeatrixUpload.id)
            filename (str): the filename to use to locate an upload in the library (FeatrixUpload.filename)
            reload (bool): if the upload isn't here, try to reload the library
        Returns:
            FeatrixUpload for the given id or filename, otherwise it will raise a FeatrixException.
        """
        if upload_id is None and filename is None:
            raise FeatrixException(
                "Must provide either an upload id or filename to locate an upload in library"
            )

        if len(self._library) == 0:
            self.get_uploads()
        if upload_id is not None:
            if str(upload_id) in self._uploads:
                return self._uploads[str(upload_id)]
            elif reload:
                self.get_uploads()
                return self.get_upload(upload_id=upload_id, reload=False)
            else:
                raise FeatrixException(f"No such file {upload_id} in library")
        if filename is not None:
            if filename in self._library:
                return self._library[filename]
            elif reload:
                self.get_uploads()
                return self.get_upload(filename=filename, reload=False)
            else:
                raise FeatrixException(f"No such file {filename} in library")

    def upload_files(
            self,
            uploads: List[pd.DataFrame | str | Path] | pd.DataFrame | str | Path,
            associate: Optional[FeatrixProject] = None,
            labels: Optional[List[str | None]] = None,
    ) -> List[FeatrixUpload]:
        """
        Upload a list of files to the Featrix system and possibly associate them with a project.
        The uploads list can be a list of filenames (by string or `pathlib.Path`) or `pandas.DataFrame objects`,
        or both.  If the `labels` list is not provided, the filenames will be used as the labels for the uploads,
        or an auto-generated label will be used for the DataFrame uploads.

        If the user passes in a project to associate, we do that as well. .

        Arguments:
            uploads: List of filenames or DataFrames to upload
            associate: a FeatrixProject to associate with
            labels: Optional list of labels to use for the uploads, if not provided, the filenames will be used for the
            upload: If `uploads` is not used, takes a single upload.

        Returns:
            List[FeatrixUpload] - the list of FeatrixUpload objects created
        """
        upload_objects = []
        if type(uploads) != list:
            uploads = [uploads]

        for idx, upload in enumerate(uploads):
            upload_objects.append(
                self.upload_file(upload, associate=associate, label=labels[idx] if labels else None)
            )
        return upload_objects

    def upload_file(
            self,
            upload: pd.DataFrame | str | Path,
            associate: Optional[FeatrixProject] = None,
            label: Optional[str] = None,
    ) -> FeatrixUpload:
        """
        Create a new upload entry in your library using either a DataFrame or the filename/Path of a CSV
        file from which to source.  It will create a new FeatrixUpload in your library, push the data from the
        DataFrame or filename to the Featrix system, and start a job that will do some post upload analysis of the
        file (typically finished in under 60 seconds).

        The Upload will be added to your library and accessible via the get_upload call.

        Additionally, If the associate argument is set to True, this upload will be associated with the current
        project.  If the associate argument is set to a FeatrixProject, it will be associated with the given project.

        Args:
            upload: (pd.DataFrame | str | Path): The data file to add to your library, referenced by a pandas DataFrame
                    or the filename (str or Path) of the CSV file to use
            associate: FeatrixProject: associate this upload with FeatrixProject
            label: Optional - use as filename if passed in a dataframe

        Returns:
            FeatrixUpload: The upload object that is created.
        """
        if isinstance(upload, pd.DataFrame):
            import tempfile
            import uuid

            td = Path(tempfile.mkdtemp())
            if label is None:
                label = f"dataframe-import-{uuid.uuid4()}.csv"
            name = td / label
            upload.to_csv(name)
            upload = FeatrixUpload.new(self, name)
            try:
                name.unlink()
                td.unlink()
            except:  # noqa
                pass
        else:
            upload = Path(upload)
            if not upload.exists():
                raise FeatrixException(f"No such file or directory {upload}")
            if not upload.is_file():
                raise FeatrixException(f"Not a file {upload}")
            upload = FeatrixUpload.new(self, str(upload))
        self._library[upload.filename] = upload
        self._uploads[upload.id] = upload
        if associate:
            if isinstance(associate, FeatrixProject):
                project = associate.associate(upload)
                self._store_project(project)
            else:
                raise FeatrixException(
                    "The associate needs to be a FeatrixProject"
                )
        return upload

    def get_neural_function(
            self,
            neural_function_id,
            in_project: Optional[FeatrixProject | str] = None
    ) -> FeatrixNeuralFunction:
        """
        Given a neural function id (and optionally, a project id), return the object that represents
        that neural function. You can then run predictions on that neural function.

        Arguments:
            neural_function_id: The id of the neural function to find
            project: Optional project to use, otherwise we walk the projects.

        Returns:
            FeatrixNeuralFunction: The model object found

        """
        projects = []
        if not in_project:
            #if not self._projects:
            self.projects()
            for project in self._projects.values():
                project.neural_functions()
                projects.append(project)
        else:
            self.projects()
            for project in self._projects.values():
                if project.id == in_project:
                    project.neural_functions()
                    projects.append(project)
            #projects = [in_project]
        project = None
        
        found_in_project = False
        for project_entry in projects:
            if in_project:
                if str(project_entry.id) == str(in_project):
                    found_in_project = True
                    model = project_entry.find_neural_function(neural_function_id)
                    if model:
                        return model
            else:
                # We try each project if no in_project was specified.
                try:
                    print(f"trying project {project_entry.id}...")
                    model = project_entry.find_neural_function(neural_function_id)
                    if model:
                        return model
                except RuntimeError:
                    # NF not found -- keep looking.
                    continue
        
        if found_in_project:
            raise ValueError(f"No neural function with id '{neural_function_id}' in project '{in_project}'")
        
        if not in_project:
            msg = f"{len(projects)} project{'s' if len(projects) != 1 else '' }"
            raise ValueError(f"No neural function with id '{neural_function_id}' found in any project in '{msg}'")

        raise ValueError(f"No neural function with id '{neural_function_id}'.")

    get_model = get_neural_function
    """
    Alias for get_neural_function
    """

    def create_embedding_space(
            self,
            project: Optional[FeatrixProject | str] = None,
            name: Optional[str] = None,
            credit_budget: int = 3,
            files: Optional[List[pd.DataFrame | str | Path]] = None,
            wait_for_completion: bool = False,
            encoder: Optional[Dict] = None,
            ignore_cols: Optional[List[str] | str] = None,
            focus_cols: Optional[List[str] | str] = None,
            **kwargs,
    ) -> "FeatrixEmbeddingSpace":  # noqa forward ref
        """
        Create a new embedding space in the project specified (FeatrixProject or
        id of a project).

        You do not need to clean nulls or make the data numeric; simply pass in strings or missing values.

        If the wait_for_completion flag is set, this will be synchronous and print periodic messages to the console
        as the embedding space is trained.  Note that the jobs are enqueued and running so if the notebook is
        interrupted, reset or crashes, the training will still complete and can be queried by using the methods later.

        In either case this returns the `FeatrixEmbeddingSpace` object. If the flag wait_for_completion was not
        set (or False), there will be a job running the training, and the caller can get that by calling the
        .get_jobs() method on the returned embedding space.

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
            FeatrixEmbeddingSpace -- the featrix model object created
        """
        from .featrix_embedding_space import FeatrixEmbeddingSpace
        from bson import ObjectId

        if project is None or (isinstance(project, str) and ObjectId.is_valid(project) is False):
            if files is None:
                raise FeatrixException(
                    f"Can not create a project named {project} and train a "
                    f"neural function without data files"
                )
            project = FeatrixProject.new(self, project if isinstance(project, str) else f"Project {name}")
        elif isinstance(project, str):
            # We know it's a ObjectId from the above test
            if project not in self._projects:
                self.projects()
            if project not in self._projects:
                raise RuntimeError(f"No such project {project}")
            project = self._projects[project]

        upload_processing_wait = wait_for_completion
        if files is not None:
            self.upload_files(files, associate=project)
            upload_processing_wait = True
            # Get the refreshed version
            project = self.get_project_by_id(project.id)

        if project.ready(wait_for_completion=upload_processing_wait) is False:
            raise FeatrixException("Project not ready for training, datafiles still being processed")
        es = FeatrixEmbeddingSpace.new_embedding_space(
            self,
            project,
            name=name,
            credit_budget=credit_budget,
            encoder=encoder,
            ignore_cols=ignore_cols,
            focus_cols=focus_cols,
            wait_for_completion=wait_for_completion,
            **kwargs
        )
        return es

    def display_embedding_explorer(
            self,
            project: Optional[FeatrixProject | str],
            embedding_space: FeatrixEmbeddingSpace = None
    ):
        """
        Not Implemented yet.

        """
        if embedding_space is None:
            if len(project._embedding_spaces_cache) == 0:
                project.embedding_spaces()
            if len(project._embedding_spaces_cache) == 0:
                raise FeatrixException(f"Project {project.name} has no embedding space trained")
            if len(project._embedding_spaces_cache) > 1:
                raise FeatrixException(f"Project {project.name} has multiple "
                                       "embedding spaces, please specify which one")
            embedding_space = project._embedding_spaces_cache[
                list(project._embedding_spaces_cache.keys())[0]
            ]
        if embedding_space.training_state != TrainingState.COMPLETED:
            raise FeatrixException(f"Embedding space training state {embedding_space.training_state} is not COMPLETED")
        explorer_data = embedding_space.get_explorer_data()
        # PLOT
        return

    def create_neural_function(
            self,
            target_fields: str | List[str],
            project: Optional[FeatrixProject | str] = None,
            credit_budget: int = 3,
            files: Optional[List[pd.DataFrame | str | Path]] = None,
            wait_for_completion: bool = False,
            encoder: Optional[Dict] = None,
            ignore_cols: Optional[List[str] | str] = None,
            focus_cols: Optional[List[str] | str] = None,
            embedding_space: Optional[FeatrixEmbeddingSpace | str] = None,
            **kwargs,
    ) -> FeatrixNeuralFunction:
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

        In either case, this will return the FeatrixNeuralFunction object that was created. If the the
        wait_for_completion flag is not set (or False), there will be 1-2 jobs that are running the training: one
        for the embedding space if it wasn't already trained, and the second for the neural function model.  They will
        be running sequentially, since the neural function is trained against the embedding space.  They can be
        retrieved by using the .get_jobs() method on the returned object.

        The caller, in the case where they do not wait for completion, can follow the progress via the jobs objects

        .. code-block:: python

           model = create_neural_function("field_name")
           training_jobs = model.get_jobs()
           for job in training_jobs:
               job = job.check()
               print(f"{job.job_type}: {job.incremental_status.message}")
               job.wait_for_completion(f"{job.id}: Training: ")

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
            FeatrixNeuralFunction, -- the featrix model that was created
        """
        from bson import ObjectId
        if project is None or (isinstance(project, str) and ObjectId.is_valid(project) is False):
            name = project if isinstance(project, str) else f"Project {target_fields}"
            project = FeatrixProject.new(self, name)
        elif isinstance(project, str):
            if project not in self._projects:
                self.projects()
            project = self._projects.get(project)
            if project is None:
                raise RuntimeError(f"No such project {project}")
        if files is not None:
            self.upload_files(files, associate=project)
            project = self.get_project_by_id(project.id)

        project = self._projects[project.id]
        if project.ready(wait_for_completion=wait_for_completion) is False:
            raise FeatrixException("Project not ready for training, datafiles still being processed")

        nf = FeatrixNeuralFunction.new_neural_function(
            self,
            project,
            target_fields,
            credit_budget,
            encoder,
            ignore_cols,
            focus_cols,
            embedding_space=embedding_space,
            **kwargs
        )
        es = FeatrixEmbeddingSpace.by_id(nf.embedding_space_id, self)
        if wait_for_completion:
            jobs = es.get_jobs() + nf.get_jobs()
            for idx, job in enumerate(jobs):
                job.wait_for_completion(f"Step {idx}/{len(jobs)}: ")
                if job.error:
                    raise FeatrixJobFailure(job)
        return nf.refresh()

    def create_explorer(
            self,
            project: Optional[FeatrixProject | str] = None,
            credit_budget: int = 3,
            files: Optional[List[pd.DataFrame | str | Path]] = None,
            wait_for_model_jobs: bool = False,
            wait_for_completion: bool = False,
            **kwargs
    ) -> Tuple[FeatrixEmbeddingSpace, FeatrixJob | List[FeatrixJob] | List[FeatrixNeuralFunction]]:
        """
        Create a new data explorer in the project.  If a project is passed in with the project
        argument, and it is of the right type (ProjectType.EXPLORER) it will be used, but if it isn't an
        explorer project, the function will raise an error.

        If the project field is a string, it can be an id of a project or it can be a name which we will use
        to create a new project.  If there is no project passed in, we will create one and generate a name.

        If an embedding space is already trained or being trained in the project, we will use that embedding space
        for the base of the explorer work, otherwise we will first train an embedding space on the data files included
        in the project.  If a list of datasets are passed into this function, we will first upload and associate
        those files with the project being used.

        With an explorer project, we must first fully train an embedding space, and then a series of jobs will be
        started to create neural function models on each column in the embedding space.

        There are two waiting modes -- wait_for_completion will wait for both the embedding space and all of the
        model jobs to be completed before returning.  In this case it will return a tuple of the EmbeddingSapce
        and a list of models associated with the explorer project.

        If this is set to false but wait_for_model_jobs is set, the function will wait unitl the embedding space is
        completed, and there are jobs for each column in place and return a tuple of the Embedding space and a
        list of Jobs associated with each model being worked on.

        If both wait_for_completion and wait_for_model_jobs are False, we will return a tuple of the Embedding space
        and the embedding space training job, unless the embedding space was already trained when the create was
        called, in which case we will revert to the return values as if wait_for_model_jobs was set (e.g.: a
        tuple of the embedding space and the model jobs).

        If either of the wait flags are set, but the notebook or script crashes during the operation, note that
        the full explorer creation procession is still operating and the caller can simpply look up the
        embedding space and call it's "get_training_jobs" and "get_model_jobs" method to inspect the progress.

        Arguments:
            project: FeatrixProject or str id of the project to use instead of self.current_project
            credit_budget(int): the default credit budget for the training
            files: a list of dataframes or paths to files to upload and associate with the project
                        (optional - if you already associated files with the project, this is redundant)
            wait_for_model_jobs(bool): wait for the model jobs to be created and return a list of these jobs.
            wait_for_completion(bool): make this synchronous, printing out status messages while waiting for the
                                    training to complete
            **kwargs:  additional arguments for the ESCreate args, if any

        Returns:
            Tuple(FeatrixNeuralFunction, Job, Job) -- the featrix model and the jobs associated with training the model
                         if wait_for_completion is True, the model returned will be fully trained, otherwise the
                         caller will need ot check on the progress of the jobs and update the model when they are
                         complete.
        """
        from bson import ObjectId
        import uuid

        if wait_for_completion:
            wait_for_model_jobs = True

        if project is None or (isinstance(project, str) and ObjectId.is_valid(project) is False):
            if files is None:
                raise FeatrixException(
                    "Can not create a project and train a neural function without data files"
                )
            name = project if isinstance(project, str) else f"Explorer project {uuid.uuid4()}"
            project = FeatrixProject.new(self, name)
        elif isinstance(project, str):
            if project not in self._projects:
                self.projects()
            project = self._projects.get(project)
            if project is None:
                raise FeatrixException(f"No such project {project}")

        if files is not None:
            self.upload_files(files, associate=project)
            project = self.get_project_by_id(project.id)

        if project.ready(wait_for_completion=wait_for_completion) is False:
            raise FeatrixException("Project not ready for training, datafiles still being processed")

        # The API will return the jobs that are in progress -- typically just the embedding space training
        # unless the embedding space is already trained, then it will return the list of model jobs (one
        # for each field)
        es, jobs = project.new_explorer(
            f"{project.name} Explorer",
            training_credits_budgeted=credit_budget,
            **kwargs
        )
        if es.training_state != TrainingState.COMPLETED:
            if wait_for_model_jobs is False:
                # Return the EmbeddingSpace and the ES Training job
                return es, jobs[0]
            # Wait for the ES training to finish, and the model jobs to be scheduled.
            # The first job will be the es training (if there is a second, it will be the wait-for-training
            # that kicks off the model creations)
            jobs[0].wait_for_completion("Embedding Space Training: ")
            es = es.by_id(es.id, self)
            tj = jobs[1] if len(jobs) > 1 and jobs[1].job_type == JobType.JOB_TYPE_ES_WAIT_TO_TRAIN else None
            jobs = es.explorer_training_jobs(wait_for_creation=True, wait_for_training_job=tj)
            # Note that we were waiting for the watcher to finish, so make sure there weren't updates to the ES
            es = es.by_id(str(es.id), self)

        # Now jobs will be the jobs of the model trainings and es will be the trained es if we get here
        if wait_for_completion:
            FeatrixJob.wait_for_jobs(self, jobs, f"Explorer Model Training ({len(jobs)} jobs)")
            models = es.neural_functions(force=True)
            return es, models
        return es, jobs

    def check_updates(self, **kwargs):
        args = {}
        for k, v in kwargs.items():
            if isinstance(v, datetime):
                args[k] = v.isoformat()
        if self.debug:
            print(f"Checking server for updates in {list(args.keys())}")
        return self.api.op("org_updates", **args)

    @classmethod
    def _check_debug(cls):
        debug = os.environ.get("FEATRIX_DEBUG", "false")
        if isinstance(debug, str) and debug.lower() in ["true", "t", "1", "on", "yes"]:
            print("Setting debug to active; warning, verbose output!")
            cls.debug = True


def new_client(
    url="https://app.featrix.com",
    client_id=None,
    client_secret=None,
    key_file=KEY_FILE_LOCATION,
    allow_unencrypted_http: bool = False,
):
    """
    Create a new Featrix Client object with the specified host and secret key.

    If the secret key is not set, the client will register for a temporary key for
    quick demos and little data sessions.

    If you want more than a temporary key, create a user account at https://featrix.ai/
    """
    return Featrix(
        url=url,
        client_id=client_id,
        client_secret=client_secret,
        key_file=key_file,
        allow_unencrypted_http=allow_unencrypted_http,
    )


if __name__ == "__main__":
    import argparse

    # for testing
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url", "-u", default="http://localhost:3001", help="URL to connect with"
    )
    ap.add_argument("--client-id", "-i", help="Client id to use")
    ap.add_argument("--client-secret", "-s", help="Client secret to use")
    ap.add_argument("--key-file", "-k", help="Path to key file to use")
    ap.add_argument(
        "--allow-http", "-a", help="allow http, set for http://localhost automatically"
    )
    args = ap.parse_args()
    if "http:" in args.url and "localhost" in args.url:
        args.allow_http = True
    fc = Featrix(
        url=args.url,
        client_id=args.client_id,
        client_secret=args.client_secret,
        key_file=args.key_file,
        allow_unencrypted_http=args.allow_http,
    )
