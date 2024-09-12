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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import ConfigDict
from pydantic import Field

from .association import UploadAssociation
from .featrix_base import FeatrixBase
from .pydantic_objectid import PydanticObjectId
from .training import TrainingState

logger = logging.getLogger(__name__)


class Model(FeatrixBase):
    # We use "model_size" -- probably need to watch future versions of pydantic since the
    # model_ namespace is protected, but turn off the warning right now.
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    name: str
    organization_id: PydanticObjectId
    project_id: Optional[PydanticObjectId] = None
    embedding_space_id: PydanticObjectId

    # FIXME: eventually we will want to allow multiple files here but most of the code probably assumes 1 right now
    associated_uploads: List[UploadAssociation] = Field(default_factory=list)

    target_columns: Optional[List[str]] = Field(default_factory=list)
    training_rows: int = 0

    training_input_columns: Optional[List[str]] = Field(default_factory=list)
    training_num_uniques: Optional[int] = None
    training_num_not_nulls: Optional[int] = None
    training_target_histogram: Optional[Dict] = None
    training_metrics: Optional[Dict] = None

    learning_rate: float = 0.0001
    epochs: int = 0
    model_size: str = "small"
    # scalar or set -- enum?
    target_type: Optional[str] = None
    mlp_predictor: Optional[str] = None

    # Job that created the model
    job_id: Optional[PydanticObjectId] = None

    training_state: TrainingState = TrainingState.UNTRAINED
    training_credits_budgeted: float = 0.0
    training_credits_actual: float = 0.0

    user_meta: Dict[str, Any] = Field(default_factory=dict)
