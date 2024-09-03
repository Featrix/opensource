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
from .training import TrainingState

logger = logging.getLogger(__name__)


class TrainingInfo(FModel):
    start_time: Optional[int | datetime] = None
    end_time: Optional[int | datetime] = None
    progress_info: Optional[Dict] = None


class EmbeddingSpace(FeatrixBase):
    model_config = ConfigDict(extra="allow")

    id: PydanticObjectId = Field(default_factory=PydanticObjectId)

    project_id: Optional[PydanticObjectId] = None

    name: str
    # Owning organization
    organization_id: PydanticObjectId

    # For now this is arbitrary meta from the system while we iterate, probably break out to a partial/full schema
    # later?
    system_meta: Dict = Field(default_factory=dict)
    user_meta: Dict = Field(default_factory=dict)

    es_neural_attrs: Optional[Dict] = None
    training_history: List[TrainingInfo] = Field(default_factory=list)

    # FIXME: Where does this come from?
    # Each dict has (at least) col_name, split_token and keep_mask
    config_split_columns: Optional[List[Dict]] = Field(default_factory=list)
    # association {
    #     id: { sample stuff }
    # }
    # Allow the user to add arbitrary notes for now. This needs to be much better...
    notes: Optional[List[str]] = None

    # Number of items in a vector (the dimension of the embedding space)
    dimension: int = 64

    training_state: TrainingState = TrainingState.UNTRAINED
    training_credits_budgeted: float = 0.0
    training_credits_actual: float = 0.0


class EmbeddingSpaceCreateRequest(FModel):
    pass


class EmbeddingSpaceCreateResponse(FModel):
    embedding_space: EmbeddingSpace


class EmbeddingSpaceDeleteRequest(FModel):
    embedding_space_id: PydanticObjectId
    force: Optional[bool] = False


class EmbeddingSpaceDeleteResponse(FModel):
    embedding_space: EmbeddingSpace


class EmbeddingSpaceEncodeRequest(FModel):
    embedding_space_id: PydanticObjectId
    upload_id: PydanticObjectId


class EmbeddingSpaceEncodeResponse(FModel):
    embedding_space_id: PydanticObjectId
    upload_id: PydanticObjectId


class EmbeddingSpaceTrainModelRequest(FModel):
    embedding_space_id: PydanticObjectId
    upload_id: PydanticObjectId
    model_id: PydanticObjectId
    # If we bring this back, enum?
    # model_size: Optional[str] = "small"
    # model_size = j.get("model_size", "small")
    target_columns: List[str]


class EmbeddingSpaceTrainModelResponse(FModel):
    pass


class EmbeddingSpaceCheckGuardrailsRequest(FModel):
    embedding_space_id: PydanticObjectId
    model_id: PydanticObjectId


class EmbeddingSpaceCheckGuardrailsResponse(FModel):
    pass


class EmbeddingSpaceValueHistogramRequest(FModel):
    embedding_space_id: PydanticObjectId
    column_name: str
    sampled_values: List[Any]


class EmbeddingSpaceValueHistogramResponse(FModel):
    pass


class EmbeddingDistanceRequest(FModel):
    embedding_space_id: PydanticObjectId
    col1: str
    col2: str


class EmbeddingDistanceResponse(FModel):
    pass
