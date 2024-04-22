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
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import ConfigDict
from pydantic import Field

from .featrix_base import FeatrixBase
from .fmodel import FModel
from .pydantic_objectid import PydanticObjectId

logger = logging.getLogger(__name__)


class Upload(FeatrixBase):
    # We have a DataFrame here, but it's not saved to the db, just used for passing around
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Customers name when they uploaded
    filename: str
    # Path name in our object store or local fs
    pathname: str

    organization_id: PydanticObjectId

    num_rows: int = Field(default=-1)
    num_cols: int = Field(default=-1)
    column_names: List[str] = Field(default_factory=list)

    # MD5 hash of file
    file_hash: str

    load_errors: Dict = Field(default_factory=dict)
    post_processing_job_id: Optional[PydanticObjectId] = None
    sample_data: List[Dict[str, Any]] = Field(default_factory=list)

    user_meta: Dict[str, Any] = Field(default_factory=dict)

    # Note, the save() will exclude this on the way to the db - it's not saved.
    df: Optional[object] = None

    processing_error_list: Optional[List[str]] = None
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None

    possible_id_columns: Optional[List[str]] = None
    detected_column_types: Optional[Dict[str, str]] = None
    ready_for_training: bool = False

    col_histograms: Optional[Dict] = None
    col_unique_counts: Optional[Dict[str, int]] = None
    col_nonnull_counts: Optional[Dict[str, int]] = None
    col_likely_positive_value: Optional[Dict[str, Any]] = None  # for binary classifier

    smart_enrichment_config: Optional[Dict[str, bool]] = None
    smart_enrichment_results: Optional[Dict] = None
    smart_enrichment_last_start_time: Optional[datetime] = None
    smart_enrichment_last_end_time: Optional[datetime] = None

    # We can have up to 3 files in an upload, where the user has pushed one file
    # and then we split things.
    test_split_ratio: float = 0.2
    train_split_ratio: float = 0.6  # actual numbers will vary
    validation_split_ratio: float = 0.2

    test_split_file_hash: Optional[str] = None
    train_split_file_hash: Optional[str] = None
    validation_split_file_hash: Optional[str] = None

    ignore_cols: Optional[List[str]] = None
    drop_duplicates: bool = True

    debug_detectors_raw: Optional[Dict] = None


class UploadFileResponse(FModel):
    upload: Upload
    post_processing_job_id: Optional[PydanticObjectId] = None
    error: Optional[str] = None


class UploadFetchUrlArgs(FModel):
    project_id: PydanticObjectId
    other_url: str
    user_name: (
        str  # not the name of the user, but instead the name the user gave to the file
    )
    meta: dict


class UploadConfigureArgs(FModel):
    # project_id: PydanticObjectId
    upload_id: PydanticObjectId
    col_name: str
    type_override: Optional[str] = None
    ignore_override: Optional[bool] = None
    # UNTIE, split stuff later.


class UploadSmartEnrichmentConfigureArgs(FModel):
    upload_id: PydanticObjectId
    email: Optional[bool] = False
    domains: Optional[bool] = False
    ipaddr: Optional[bool] = False
    timestamps: Optional[bool] = False
