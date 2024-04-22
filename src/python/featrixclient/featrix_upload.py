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

from pydantic import Field

from .api_urls import ApiInfo
from .models.upload import Upload


#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Featrix, Inc, All Rights Reserved
#
#  Proprietary and Confidential.  Unauthorized use, copying or dissemination
#  of these materials is strictly prohibited.


class FeatrixUpload(Upload):
    fc: Optional[Any] = Field(default=None, exclude=True)

    @classmethod
    def new(
        cls, fc: Any, filename: str, user_meta: Optional[Dict] = None
    ) -> "FeatrixUpload":
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"{filename} does not exist")
        upload = fc.api.op(
            "uploads_create", **{"file": (path.name, path.open("r"), "text/csv")}
        )
        return ApiInfo.reclass(cls, upload, fc=fc)

    @classmethod
    def all(cls, fc: Any) -> List["FeatrixUpload"]:
        results = fc.api.op("uploads_get_all")
        return ApiInfo.reclass(cls, results, fc=fc)

    @classmethod
    def by_id(cls, fc: Any, upload_id: str) -> "FeatrixUpload":
        results = fc.api.op("uploads_get", upload_id=upload_id)
        return ApiInfo.reclass(cls, results, fc=fc)

    @classmethod
    def by_hash(cls, fc: Any, hash_id: str) -> "FeatrixUpload":
        results = fc.api.op("uploads_get_by_hash", hash_id=hash_id)
        return ApiInfo.reclass(cls, results, fc=fc)

    def delete(self) -> "FeatrixUpload":
        results = self.fc.api.op("uploads_delete", upload_id=self.id)
        return ApiInfo.reclass(FeatrixUpload, results, fc=self.fc)

    def jobs(self):
        results = self.fc.api.op("uploads_get_jobs", upload_id=self.id)
        return ApiInfo.reclass(FeatrixUpload, results, fc=self.fc)
