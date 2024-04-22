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
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import pandas as pd
from .exceptions import FeatrixException

from .api import FeatrixApi
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
    def current_project(self, value: FeatrixProject) -> None:
        """
        Set the current project, which should be a FeatrixProject in the self.projects list.

        Args:
            value (FeatrixProject): the project to make the current project

        """
        if not isinstance(value, FeatrixProject):
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

    def get_upload(self, upload_id: str = None, filename: str = None) -> FeatrixUpload:
        """
        Return the upload object (FeatrixUpload) for teh given upload id or filename.

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

    def upload_file(
        self,
        upload: pd.DataFrame | str | Path,
        associate: bool | FeatrixProject = False,
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
        Returns:
            FeatrixUpload: The upload object that is created.
        """
        if isinstance(upload, pd.DataFrame):
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as _f:
                upload.to_csv(_f)
                name = _f.name
            upload = FeatrixUpload.new(self, name)
            Path(name).unlink()
        else:
            upload = Path(upload)
            if not upload.exists():
                raise FeatrixException(f"No such file or directory {upload}")
            if not upload.is_file():
                raise FeatrixException(f"Not a file {upload}")
            upload = FeatrixUpload.new(self, str(upload))
        self._library[upload.name] = upload
        self._uploads[upload.id] = upload
        if associate:
            if isinstance(associate, FeatrixProject):
                associate.associate(upload)
            elif associate is True and self._current_project is not None:
                self._current_project.associate(upload)
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
