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

from datetime import datetime
from enum import Enum
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from .featrix_base import FeatrixBase
from .fmodel import FModel


class Features(Enum):
    Demo = "demo"
    Funnel = "funnel"
    Root = "root"  # Featrix main org for easy access


class Organization(FeatrixBase):
    name: str

    # The ID of this org in HubSpot (if it's been added/sync'ed yet)
    hubspot_id: Optional[str] = None

    # Extra features the org is entitled to
    features: List[str] = Field(default_factory=list)

    # If the organization has a private bucket for storage
    storage_bucket: Optional[str] = None
    # If the organization is storing things encrypted, this is the public key
    encryption_public_key: Optional[str] = None


class OrganizationBrief(BaseModel):
    name: str
    storage_bucket: Optional[str] = None
    encryption_public_key: Optional[str] = None


class UpdatedRequest(FModel):
    activity: Optional[datetime] = None
    upload: Optional[datetime] = None
    project: Optional[datetime] = None
    embedding_space: Optional[datetime] = None
    model: Optional[datetime] = None
    prediction: Optional[datetime] = None
    members: Optional[datetime] = None
    api_key: Optional[datetime] = None
    invitation: Optional[datetime] = None
    job_meta: Optional[datetime] = None


class UpdatedResponse(FModel):
    activity: Optional[bool] = None
    upload: Optional[bool] = None
    project: Optional[bool] = None
    embedding_space: Optional[bool] = None
    model: Optional[bool] = None
    prediction: Optional[bool] = None
    members: Optional[bool] = None
    api_key: Optional[bool] = None
    invitation: Optional[bool] = None
    job_meta: Optional[bool] = None
