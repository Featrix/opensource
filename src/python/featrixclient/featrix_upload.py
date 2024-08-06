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

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import Field, PrivateAttr

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .models.upload import Upload
from .config import settings


class FeatrixUpload(Upload):
    """
    Represents a file upload to Featrix server for use in training.  This is the metadata of the file, including
    the results of Featrix's analysis of the file and possible enrichments.
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
    def by_id(cls, upload_id: str, fc: Any) -> "FeatrixUpload":
        """
        Get a specific upload by its id

        Args:
            upload_id: str: the upload id
            fc: Featrix class instance

        Returns:
            FeatrixUpload: The upload if it exists, otherwise None
        """
        results = fc.api.op("uploads_get", upload_id=upload_id)
        return ApiInfo.reclass(cls, results, fc=fc)

    @classmethod
    def by_hash(cls, hash_id: str, fc: Any) -> "FeatrixUpload":
        """
        Get a specific upload by its hash

        Args:
            hash_id: str: the hash id
            fc: Featrix class instance

        Returns:
            FeatrixUpload: The upload if it exists, otherwise None
        """
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
