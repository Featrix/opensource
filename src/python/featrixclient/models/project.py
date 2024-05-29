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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .association import UploadAssociation
from .featrix_base import FeatrixBase
from .fmodel import FModel
from .pydantic_objectid import PydanticObjectId


class ProjectRowMetaData(FModel):
    row_idx_start: int
    num_rows: int
    columns_list: List[str] = Field(default_factory=list)
    label: str
    debug_index: int

    def contains_row(self, idx):
        return self.row_idx_start <= idx < self.row_idx_start + self.num_rows


class ProjectTag(Enum):
    Funnel = "funnel"
    Spatial = "spatial"


class ProjectFieldMapping(BaseModel):
    target_field: str
    source_field: str


class ProjectTableMapping(BaseModel):
    target: str  # filename?
    source: str  # filename?
    fields: list[ProjectFieldMapping]


class ProjectType(Enum):
    SDK = "sdk"
    HAYSTACK = "haystack"
    EXPLORER = "explorer"


class InvoiceState(Enum):
    CREATED = "created"
    PAID = "paid"
    CANCELLED = "cancelled"


class ProjectInvoiceRecord(BaseModel):
    stripe_invoice_id: str
    state: InvoiceState
    date: datetime
    training_complete: bool


class Project(FeatrixBase):
    model_config = ConfigDict(extra="allow")

    has_haystack: bool = False
    has_feeds: bool = False

    name: str
    organization_id: PydanticObjectId
    # project_type: ProjectType
    tags: List[str] = Field(default_factory=list)

    type: ProjectType = Field(default=ProjectType.SDK)

    # List of uploads that are associated with this project
    associated_uploads: List[UploadAssociation] = Field(default_factory=list)
    embedding_space_ids: List[PydanticObjectId] = Field(default_factory=list)
    # self.get_ignore_cols -> self.ignore_cols
    ignore_cols: List[str] = Field(default_factory=list)
    focus_cols: List[str]  = Field(default_factory=list)
    type_overrides: List[str] = Field(default_factory=list)

    mappings: list[ProjectTableMapping] = []
    user_meta: Dict[str, Any] = Field(default_factory=dict)

    readme_text: Optional[str] = None  # = Field(default_factory=str)
    banner_text: Optional[str] = None


class ProjectCreateRequest(FModel):
    name: Optional[str]
    tags: Optional[List[str]] = []
    # User meta that can be submitted and enriches the Dataspace object in mongo
    user_meta: Optional[Dict[str, Any]] = {}


class ProjectDeleteResponse(FModel):
    project: Project


class ProjectUnassociateRequest(FModel):
    project_id: PydanticObjectId
    upload_id: PydanticObjectId


class ProjectAssociateRequest(FModel):
    project_id: PydanticObjectId
    upload_id: PydanticObjectId
    label: Optional[str] = None
    sample_row_count: Optional[int] = 0
    sample_percentage: Optional[float] = 1
    drop_duplicates: Optional[bool] = True


class AllFieldsResponse(BaseModel):
    name: str
    file: str
    short_name: str


class ProjectEditRequest(BaseModel):
    project_id: PydanticObjectId
    name: str


class ProjectIgnoreColsRequest(FModel):
    project_id: PydanticObjectId
    columns: List[str]


class ProjectFocusColsRequest(FModel):
    project_id: PydanticObjectId
    columns: List[str]


class ProjectSearchMappingsRequest(FModel):
    project_id: PydanticObjectId
    ignore_cols: bool


class ProjectAddMappingsRequest(FModel):
    project_id: PydanticObjectId
    mappings: ProjectTableMapping


class ProjectRemoveMappingsRequest(FModel):
    project_id: PydanticObjectId
    mappings: ProjectTableMapping


class ProjectAutoJoinRequest(FModel):
    project_id: PydanticObjectId
    # vector_space_id: PydanticObjectId


class ProjectCreateParameters(BaseModel):
    type: ProjectTag
    organization_id: PydanticObjectId | None = None


class ProjectWithMetadata(BaseModel):
    name: str
    organization_id: PydanticObjectId
    project_type: ProjectTag
    metadata: dict
