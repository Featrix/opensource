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

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import PrivateAttr

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .models import PydanticObjectId
from .models.upload import Upload


class FeatrixUpload(Upload):
    """
    Represents a file upload to the Featrix server for training, including metadata, analysis results, and possible enrichments.

    Retrieve an upload by ID with `.by_id()`, and refresh an existing upload object with `.refresh()` if it might have changed.
    """


    _fc: Optional[Any] = PrivateAttr(default=None)
    """Reference to the Featrix class  that retrieved or created this project, used for API calls/credentials"""

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

    def get_jobs(self, active: bool = True) -> List["FeatrixJob"]:  # noqa forward ref
        """
        Return a list of jobs associated with this upload.

        By default, only active jobs are returned. Use the arguments to filter results.

        Args:
            active (bool): If `True`, return only active jobs.
            training (bool): If `True`, return only training jobs.

        Returns:
            List[FeatrixJob]: The list of jobs associated with this upload.
        """

        from .featrix_job import FeatrixJob  # noqa forward ref

        jobs = []
        for job in FeatrixJob.by_upload(self):
            if active and job.finished:
                continue
            jobs.append(job)
        return jobs

    @classmethod
    def new(
        cls, fc: Any, filename: str | Path, user_meta: Optional[Dict] = None
    ) -> "FeatrixUpload":
        """
        Create a new FeatrixUpload object and upload the file to the server.
        """
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"{filename} does not exist")
        upload = fc.api.op(
            "uploads_create", **{"file": (path.name, path.open("r"), "text/csv")}
        )
        return ApiInfo.reclass(cls, upload, fc=fc)

    @classmethod
    def all(cls, fc: Any) -> List["FeatrixUpload"]:
        """
        Get all uploads on the server

        Args:
            fc: Featrix class instance

        Returns:
            List[FeatrixUpload]: List of all uploads on the server
        """
        results = fc.api.op("uploads_get_all")
        return ApiInfo.reclass(cls, results, fc=fc)

    @classmethod
    def by_id(
        cls,
        upload_id: str | PydanticObjectId,
        fc: Optional["Featrix"] = None,  # noqa F821
    ) -> "FeatrixUpload":
        """
        Get a specific upload by its id

        Args:
            upload_id: str: the upload id
            fc: Featrix class instance

        Returns:
            FeatrixUpload: The upload if it exists, otherwise None
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()
        results = fc.api.op("uploads_get", upload_id=str(upload_id))
        return ApiInfo.reclass(cls, results, fc=fc)

    @classmethod
    def by_hash(cls, hash_id: str, fc: Optional["Featrix"] = None) -> "FeatrixUpload":  # noqa F821
        """
        Get a specific upload by its hash

        Args:
            hash_id: str: the hash id
            fc: Featrix class instance

        Returns:
            FeatrixUpload: The upload if it exists, otherwise None
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()
        results = fc.api.op("uploads_get_by_hash", hash_id=hash_id)
        return ApiInfo.reclass(cls, results, fc=fc)

    def delete(self) -> "FeatrixUpload":
        """
        Delete the upload from the server

        Returns:
            FeatrixUpload: The upload that was deleted
        """
        results = self._fc.api.op("uploads_delete", upload_id=self.id)
        return ApiInfo.reclass(FeatrixUpload, results, fc=self._fc)

    def jobs(self):
        """
        Get the jobs associated with this upload

        Returns:
            List[FeatrixJob]: List of jobs associated with this upload
        """
        results = self._fc.api.op("uploads_get_jobs", upload_id=self.id)
        return ApiInfo.reclass(FeatrixUpload, results, fc=self._fc)
