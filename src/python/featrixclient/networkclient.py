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
from featrixclient.models import TrainingState, JobType, PydanticObjectId
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
    instance: "Featrix" = None

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

        This requires authentication using API Keys (client ID and secret) from the Featrix UI at https://app.featrix.com. The credentials can be provided in one of three ways:

        1) As `client_id` and `client_secret` arguments.
        2) Set in the environment variables `FEATRIX_CLIENT_ID` and `FEATRIX_CLIENT_SECRET`.
        3) Stored in `${HOME}/.featrix.key` (or a specified file via `key_file`), with the format:

        FEATRIX_CLIENT_ID=xxxxxxxxxx
        FEATRIX_CLIENT_SECRET=xxxxxxxxxx

        To generate an API Key, register and log in at https://app.featrix.com/. Under your Profile menu, select "Manage API Keys."

        Args:
            url (str): URL of the Featrix server (default: https://app.featrix.com/).
            client_id (str | None): API Key client ID.
            client_secret (str | None): API Key client secret.
            key_file (str | Path | None): File containing API Key credentials.
            allow_unencrypted_http (bool): Allow non-HTTPS connections (for development).

        Returns:
            A Featrix object for accessing neural functions and performing predictive AI queries.
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
        Featrix.instance = self

    @staticmethod
    def get_instance() -> "Featrix":
        return Featrix.instance

    def _store_project(self, project: FeatrixProject):
        self._projects[str(project.id)] = project

    def projects(self) -> List[FeatrixProject]:
        """
        Return a list of all projects in your account.

        Each project includes its name, associated data sets (`associated_uploads`), any mappings between data sets (`mappings`), and columns ignored by the project (`ignore_cols`).

        Returns:
            List[FeatrixProject]: A list of projects.
        """
        projects = FeatrixProject.all(self)
        for project in projects:
            if self.debug:
                print(f"Found project: {project.model_dump_json(indent=4)}")
            self._store_project(project)
        return projects

    def get_project_by_id(
        self, project_id: str | PydanticObjectId
    ) -> FeatrixProject | None:
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

    def get_project_by_name(
        self, name: str
    ) -> FeatrixProject | List[FeatrixProject] | None:
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
        Create a new project and set it as the current project.

        Args:
            name (str | None): Optional project name; auto-named if not provided.
            user_meta (dict | None): Optional metadata to associate with the project.
            tags (list | None): Optional tags to associate with the project.

        Returns:
            FeatrixProject: The newly created project object.
        """
        project = FeatrixProject.new(self, name, user_meta=user_meta, tags=tags)
        self._store_project(project)
        return project

    # def drop_project(self, project: FeatrixProject | str) -> None:
    #     """
    #     Drop a project from the cache of projects.  This will not delete the project from the Featrix system,
    #     just remove it from the cache of projects in the Featrix object.

    #     Args:
    #         project: The project object or the id of the project to drop from the cache
    #     """
    #     project_id = str(project.id) if isinstance(project, FeatrixProject) else project
    #     if project_id in self._projects:
    #         del self._projects[project_id]

    def get_uploads(self) -> None:
        """
        Get all the FeatrixUpload entries that describe files the user has uploaded to the Featrix system.
        """
        uploads = FeatrixUpload.all(self)
        for upload in uploads:
            self._library[upload.filename] = upload
            self._uploads[str(upload.id)] = upload

    def get_upload(
        self, upload_id: str = None, filename: str = None, reload: bool = True
    ) -> FeatrixUpload:
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
        Upload files or DataFrames to the Featrix system, optionally associating them with a project.

        Args:
            uploads (list): List of filenames (str or `pathlib.Path`) or `pandas.DataFrame` objects to upload.
            associate (FeatrixProject | None): Optional project to associate with the uploads.
            labels (list | None): Optional labels for the uploads; defaults to filenames or auto-generated labels for DataFrames.
            upload: If `uploads` is not used, takes a single upload.

        Returns:
            List[FeatrixUpload]: The list of created FeatrixUpload objects.
        """
        upload_objects = []
        if uploads is not list:
            uploads = [uploads]

        for idx, upload in enumerate(uploads):
            upload_objects.append(
                self.upload_file(
                    upload, associate=associate, label=labels[idx] if labels else None
                )
            )
        return upload_objects

    def upload_file(
        self,
        upload: pd.DataFrame | str | Path,
        associate: Optional[FeatrixProject] = None,
        label: Optional[str] = None,
    ) -> FeatrixUpload:
        """
        Create a new upload in your library from a DataFrame or CSV file.

        This creates a `FeatrixUpload` in your library, uploads the data to Featrix, and starts a post-upload analysis (typically completed within 60 seconds). The upload will be accessible via `get_upload`.

        If `associate` is `True`, the upload is associated with the current project. If `associate` is a `FeatrixProject`, it will be associated with that specific project.

        Args:
            upload (pd.DataFrame | str | Path): The data to upload, either as a DataFrame or a CSV file path.
            associate (FeatrixProject | bool | None): Optionally associate the upload with a project.
            label (str | None): Optional label for the upload; used as filename if provided with a DataFrame.

        Returns:
            FeatrixUpload: The created upload object.
        """
        if isinstance(upload, pd.DataFrame):
            import tempfile
            import uuid

            td = Path(tempfile.mkdtemp())
            if label is None:
                label = f"dataframe-import-{uuid.uuid4()}.csv"
            name = td / label
            upload.to_csv(name, index=None)
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
                raise FeatrixException("The associate needs to be a FeatrixProject")
        return upload

    def get_neural_function(
        self, neural_function_id, in_project: Optional[FeatrixProject | str] = None
    ) -> FeatrixNeuralFunction:
        """
        Retrieve a neural function object by its ID, optionally within a specific project.

        Args:
            neural_function_id (str): The ID of the neural function to retrieve.
            project (FeatrixProject | None): Optional project to search within; otherwise, all projects are searched.

        Returns:
            FeatrixNeuralFunction: The retrieved neural function object.
        """
        projects = []
        if not in_project:
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
                    # print(f"trying project {project_entry.id}...")
                    model = project_entry.find_neural_function(neural_function_id)
                    if model:
                        return model
                except RuntimeError:
                    # NF not found -- keep looking.
                    continue

        if found_in_project:
            raise ValueError(
                f"No neural function with id '{neural_function_id}' in project '{in_project}'"
            )

        if not in_project:
            msg = f"{len(projects)} project{'s' if len(projects) != 1 else '' }"
            raise ValueError(
                f"No neural function with id '{neural_function_id}' found in any project in '{msg}'"
            )

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
        Create a new embedding space within a specified project.

        You can pass in strings or missing values without cleaning the data. If `wait_for_completion` is `True`, the process will be synchronous, with periodic status messages. Even if interrupted, the training will complete and can be queried later.

        Args:
            project (FeatrixProject | str | None): The project to use (ID or object); a new project is created if not provided.
            name (str): The name of the embedding space.
            credit_budget (int): The credit budget for training.
            files (list | None): Optional list of DataFrames or file paths to upload and associate with the project.
            wait_for_completion (bool): If `True`, runs synchronously, printing status messages during training.
            encoder (dict | None): Optional dictionary of encoder overrides.
            ignore_cols (list | str | None): Optional columns to ignore during training (list or comma-separated string).
            focus_cols (list | str | None): Optional columns to focus on during training (list or comma-separated string).
            **kwargs: Additional arguments for `ESCreateArgs`, e.g., `rows=1000`.

        Returns:
            FeatrixEmbeddingSpace: The created embedding space object.
        """
        from .featrix_embedding_space import FeatrixEmbeddingSpace
        from bson import ObjectId

        if project is None or (
            isinstance(project, str) and ObjectId.is_valid(project) is False
        ):
            if files is None:
                raise FeatrixException(
                    f"Can not create a project named {project} and train a neural function without data files"
                )
            project = FeatrixProject.new(
                self, project if isinstance(project, str) else f"Project {name}"
            )
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
            raise FeatrixException(
                "Project not ready for training, datafiles still being processed"
            )
        es = FeatrixEmbeddingSpace.new_embedding_space(
            self,
            project,
            name=name,
            credit_budget=credit_budget,
            encoder=encoder,
            ignore_cols=ignore_cols,
            focus_cols=focus_cols,
            wait_for_completion=wait_for_completion,
            **kwargs,
        )
        return es

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
        Create a new neural function within a specified project.

        If a project (ID or `FeatrixProject`) is provided, it will be used. If it's a string and not an ID, a new project will be created with that name (or a name derived from `target_fields` if none is provided).

        If an embedding space exists or is being trained in the project, it will be used to train the neural function. Otherwise, an embedding space will first be trained. If datasets are provided, they will be uploaded and associated with the project.

        If `wait_for_completion` is `True`, the process will be synchronous, with status updates. The training jobs will complete even if interrupted and can be queried later. The method returns the created `FeatrixNeuralFunction`. If not waiting for completion, training jobs can be monitored via `.get_jobs()`.

        Args:
            target_fields (str | list): Field name(s) to target in predictions.
            project (FeatrixProject | str | None): Project to use, or name for a new project.
            credit_budget (int): Credit budget for training.
            files (list | None): DataFrames or file paths to upload and associate with the project (optional).
            wait_for_completion (bool): Run synchronously with status updates.
            encoder (dict | None): Optional encoder overrides for the embedding space.
            ignore_cols (list | str | None): Columns to ignore in training (list or comma-separated string).
            focus_cols (list | str | None): Columns to focus on in training (list or comma-separated string).
            **kwargs: Additional arguments for `ESCreateArgs`, e.g., `rows=1000`.

        Returns:
            FeatrixNeuralFunction: The created neural function model.
        """
        from bson import ObjectId

        if project is None or (
            isinstance(project, str) and ObjectId.is_valid(project) is False
        ):
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
            raise FeatrixException(
                "Project not ready for training, datafiles still being processed"
            )
        project = project.refresh()

        nf = FeatrixNeuralFunction.new_neural_function(
            self,
            project,
            target_fields,
            credit_budget,
            encoder,
            ignore_cols,
            focus_cols,
            embedding_space=embedding_space,
            **kwargs,
        )
        es = FeatrixEmbeddingSpace.by_id(nf.embedding_space_id, self)
        if wait_for_completion:
            jobs = es.get_jobs() + nf.get_jobs()
            for idx, job in enumerate(jobs):
                job.wait_for_completion(f"Step {idx}/{len(jobs)}: ")
                if job.error:
                    raise FeatrixJobFailure(job)
        return nf.refresh()


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
        "--url", "-u", default="http://localhost:3001/", help="URL to connect with"
    )
    ap.add_argument("--client-id", "-i", help="Client id to use")
    ap.add_argument("--client-secret", "-s", help="Client secret to use")
    ap.add_argument("--key-file", "-k", help="Path to key file to use")
    ap.add_argument(
        "--allow-http", "-a", help="allow http, set for http://localhost/ automatically"
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
