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
from featrixclient.models import TrainingState
from .api import FeatrixApi
from .exceptions import FeatrixException, FeatrixJobFailure
from .featrix_model import FeatrixModel
from .featrix_project import FeatrixProject
from .featrix_upload import FeatrixUpload
from .version import version, publish_host, publish_time

__version__ = f"{version}: published from {publish_host} at {publish_time}"

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

        The caller has to provide authentication credentials. These are API Keys, generated on the Featrix website
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
        self._current_project = None
        self._current_model = None
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
            self._projects[str(project.id)] = project
        if self.current_project is None:
            if len(projects) > 0:
                self.current_project = projects[0]
        return projects

    @property
    def current_project(self) -> FeatrixProject:
        """
        Return the current project which is being used by the class.
        """
        return self._current_project

    @current_project.setter
    def current_project(self, value: FeatrixProject | str) -> None:
        """
        Set the current project, which should be a FeatrixProject in the self.projects list.

        Args:
            value (FeatrixProject): the project to make the current project

        """
        project = None
        if isinstance(value, str):
            # Should be a project id
            if value not in self._projects:
                self.projects()
            if value not in self._projects:
                raise FeatrixException(
                    f"Project id {value} not found in organizations project list."
                )
            project = self._projects[value]
        elif isinstance(value, FeatrixProject):
            project = value
        elif value is None:
            project = self.current_project
        if project is None:
            raise FeatrixException(
                "Can only set the current project to a valid FeatrixProject class"
            )
        self._current_project = value
        if self.debug:
            print(
                f"Setting current project to {self._current_project.name} - {self._current_project.id}"
            )
            print(
                f"Refreshing embedding spaces for project {self._current_project.name}"
            )
        self._current_project.embedding_spaces()

    def create_project(
        self,
        name: Optional[str] = None,
        user_meta: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Create a new project and make it the current project using the name provided.
        """
        self.current_project = FeatrixProject.new(self, name, user_meta=user_meta, tags=tags)
        return self.current_project

    def get_upload(self, upload_id: str = None, filename: str = None) -> FeatrixUpload:
        """
        Return the upload object (FeatrixUpload) for the given upload id or filename.

        Args:
            upload_id (str): Upload id of an upload to find in the library (FeatrixUpload.id)
            filename (str): the filename to use to locate an upload in the library (FeatrixUpload.filename)

        Returns:
            FeatrixUpload for the given id or filename, otherwise it will raise a FeatrixException.
        """
        if upload_id is None and filename is None:
            raise FeatrixException(
                "Must provide either an upload id or filename to locate an upload in library"
            )

        if len(self._library) == 0:
            uploads = FeatrixUpload.all(self)
            for upload in uploads:
                self._library[upload.filename] = upload
                self._uploads[str(upload.id)] = upload
        if upload_id is not None:
            if str(upload_id) in self._uploads:
                return self._uploads[str(upload_id)]
            else:
                raise FeatrixException(f"No such file {upload_id} in library")
        if filename is not None:
            if filename in self._library:
                return self._library[filename]
            else:
                raise FeatrixException(f"No such file {filename} in library")

    def upload_files(
            self,
            uploads: List[pd.DataFrame | str | Path],
            associate: bool | FeatrixProject = False,
            labels: Optional[List[str | None]] = None,
    ):
        upload_objects = []
        for idx, upload in enumerate(uploads):
            upload_objects.append(
                self.upload_file(upload, associate=associate, label=labels[idx] if labels else None)
            )
        return upload_objects

    def upload_file(
            self,
            upload: pd.DataFrame | str | Path,
            associate: bool | FeatrixProject = False,
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
            associate: (bool | FeatrixProject): IF set to true, associate this upload with the current project,
                    if the field is set to a FeatrixProject, associate it with that specific project.
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
                self._projects[str(project.id)] = project
            elif associate is True and self._current_project is not None:
                self.current_project = self._current_project.associate(upload)
            else:
                raise FeatrixException(
                    "No current project with which to associate upload"
                )
        return upload

    @property
    def current_model(self) -> FeatrixModel:
        """
        Get the current active model/neural function, if there is one.

        Returns:
            FeatrixModel | None: Current active model
        """
        return self._current_model

    @current_model.setter
    def current_model(self, value: FeatrixModel) -> None:
        """
        Set the current active model/n
        Parameters
        ----------

        Returns
        -------

        """
        if not isinstance(value, FeatrixModel):
            raise FeatrixException("Value must be a FeatrixModel class")
        if self._current_model != value:
            self._current_model = value

    @property
    def current_neural_function(self):
        """
        Get teh current active neural function (aka model), if there is one.

        Returns
        -------
            FeatrixModel | None: Current active model
        """
        return self.current_model

    @current_neural_function.setter
    def current_neural_function(self, value: FeatrixModel) -> None:
        """
        Set the current neural function to operate on.

        Parameters
        ----------
        value : FeatrixModel

        """
        self.current_model = value

    def neural_functions(
        self, project: Optional[FeatrixProject | str] = None
    ) -> List[FeatrixModel]:
        """
        Convenience method for getting a project's list of models.  It is quicker to know which
        embedding space contains the model and get at it that way if the project contains multiple embedding
        spaces, but for the projects with just a single embedding space, this is easier to use.
        Method will use the current project if there is no project argument, and will load projects if they
        aren't already in place.

        Args:
            project: FeatrixProject or str: The FeatrixProject to or the id of the FeatrixProject to use instead
                        of current_project

        Returns:
            List[FeatrixModel] List of Neural Functions (FeatrixModel) objects
        """
        if project is None:
            project = self.current_project
        if project is None:
            self.projects()
        if project is None:
            raise FeatrixException("There are no projects available.")
        return project.models()

    models = neural_functions

    def get_neural_function(
        self, neural_function_id, project: Optional[FeatrixProject | str] = None
    ) -> FeatrixModel:
        """

        Parameters
        ----------

        Returns
        -------

        """
        if project is not None:
            self.current_project = (
                project
                if isinstance(project, FeatrixProject)
                else self._projects[project]
            )
        if len(self._projects) == 0:
            self.projects()
            self.models()
        if self.current_project is None:
            raise FeatrixException("There are no projects in this account")

        model = self.current_project.find_model(neural_function_id)
        if model is None:
            if self.debug:
                print(
                    f"Model {neural_function_id} not found in current project "
                    f"{self.current_project.id}, checking others"
                )
            # Just deal with multi projects for now -- need a better way
            for p in self._projects.values():
                self.current_project = p
                self.models()
                model = self.current_project.find_model(neural_function_id)
                if model:
                    break
            else:
                raise ValueError(f"No Neural function model id {neural_function_id}")
        if self.debug:
            print(f"Found neural function/model {model.name} - {model.id}")
        self.current_model = model
        return self.current_model

    get_model = get_neural_function

    def create_embedding_space(
            self,
            project: Optional[FeatrixProject | str] = None,
            name: Optional[str] = None,
            credit_budget: int = 3,
            files: Optional[List[pd.DataFrame | str | Path]] = None,
            wait_for_completion: bool = False
    ) -> Tuple["FeatrixEmbeddingSpace", FeatrixJob]:  # noqa forward ref
        """


        Arguments:
            project: FeatrixProject or str id of the project to use instead of self.current_project
            name: str -- name of embedding space
            credit_budget(int): the default credit budget for the training
            files: a list of dataframes or paths to files to upload and associate with the project
                        (optional - if you already associated files with the project, this is redundant)
            wait_for_completion(bool): make this synchronous, printing out status messages while waiting for the
                                    training to complete

        Returns:
            Tuple(FeatrixEmbeddingSpace, Job) -- the featrix model and the jobs associated with training the model
                         if wait_for_completion is True, the model returned will be fully trained, otherwise the
                         caller will need ot check on the progress of the jobs and update the model when they are
                         complete.
        """
        from .featrix_embedding_space import FeatrixEmbeddingSpace

        if project is not None:
            try:
                self.current_project = project
            except FeatrixException:
                from bson import ObjectId

                if ObjectId.is_valid(project):
                    raise
                else:
                    if files is None:
                        raise FeatrixException(
                            f"Can not create a project named {project} and train a "
                            f"neural function without data files"
                        )
                    self.current_project = FeatrixProject.new(self, project)

        upload_processing_wait = wait_for_completion
        if files is not None:
            self.upload_files(files, associate=True)
            upload_processing_wait = True

        if self.current_project.ready(wait_for_completion=upload_processing_wait) is False:
            raise FeatrixException("Project not ready for training, datafiles still being processed")
        es, job = FeatrixEmbeddingSpace.new_embedding_space(
            self,
            name=name,
            credit_budget=credit_budget,
        )
        if wait_for_completion:
            job = job.wait_for_completion("Training Embedding Space: ")
        if job.error:
            raise FeatrixJobFailure(job)
        es = FeatrixEmbeddingSpace.by_id(job.embedding_space_id, self)
        return es, job

    def display_embedding_explorer(
            self,
            project: Optional[FeatrixProject | str] = None,
            embedding_space: FeatrixEmbeddingSpace = None
    ):
        if project is not None:
            self.current_project = project
        if embedding_space is None:
            if len(self.current_project.embedding_spaces_cache) == 0:
                self.current_project.embedding_spaces()
            if len(self.current_project.embedding_spaces_cache) == 0:
                raise FeatrixException(f"Project {self.current_project.name} has no embedding space trained")
            if len(self.current_project.embedding_spaces_cache) > 1:
                raise FeatrixException(f"Project {self.current_project.name} has multiple "
                                       "embedding spaces, please specify which one")
            embedding_space = self.current_project.embedding_spaces_cache[
                list(self.current_project.embedding_spaces_cache.keys())[0]
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
            wait_for_completion: bool = False
    ) -> Tuple[FeatrixModel, FeatrixJob, FeatrixJob]:
        """
        Create a new neural function in the current project.  If a project is passed in with the project_or_id
        argument, it is made the current project.  If the project_or_id is a string, if it is an object id
        (standard id for a FeatrixProject) we attempt to find that project and make it the current project.  If the
        string in project_or_id is not a valid object id, we assume it is a name to be used for creating a new
        project.

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
            ```
            model, es_training_job, nf_training_job = create_neural_function("field_name")
            if nf_training_job.completed is False:
                nf_training_job = nf_training_job.check()
                print(nf_training_job.incremental_status)
            ```

        They can also just wait on the neural function model's field training_state to be set to
        TrainingState.COMPLETED ("trained")

        Arguments:
            target_fields: the field name(s) to target in the prediction
            project: FeatrixProject or str id of the project to use instead of self.current_project
            credit_budget(int): the default credit budget for the training
            files: a list of dataframes or paths to files to upload and associate with the project
                        (optional - if you already associated files with the project, this is redundant)
            wait_for_completion(bool): make this synchronous, printing out status messages while waiting for the
                                    training to complete

        Returns:
            Tuple(FeatrixModel, Job, Job) -- the featrix model and the jobs associated with training the model
                         if wait_for_completion is True, the model returned will be fully trained, otherwise the
                         caller will need ot check on the progress of the jobs and update the model when they are
                         complete.
        """
        if project is not None:
            try:
                self.current_project = project
            except FeatrixException:
                from bson import ObjectId

                if ObjectId.is_valid(project):
                    raise
                else:
                    if files is None:
                        raise FeatrixException(
                            f"Can not create a project named {project} and train a "
                            f"neural function without data files"
                        )
                    self.current_project = FeatrixProject.new(self, project)
            except Exception as e:
                print(f"Exception {e} happened!")
                raise

        if files is not None:
            self.upload_files(files, associate=True)

        if self.current_project.ready(wait_for_completion=wait_for_completion) is False:
            raise FeatrixException("Project not ready for training, datafiles still being processed")

        jobs = FeatrixModel.new_neural_function(self, target_fields, credit_budget)
        if wait_for_completion:
            # If we are leveraging an embedding space we created previously, the job will be marked as finished already
            fini = []
            if jobs[0].finished is False:
                fini.append(jobs[0].wait_for_completion("Step 1/2: "))
            fini.append(jobs[1].wait_for_completion("Step 2/2: "))
            jobs = fini
        for job in jobs:
            if job.error:
                raise FeatrixJobFailure(job)
        model = FeatrixModel.from_job(jobs[1])
        return model, jobs[0], jobs[1]

    def predictions(self):
        if self.current_model is None:
            raise FeatrixException(
                "There is no current model, can not retrieve past predictions"
            )
        return self.current_model.predictions()

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
