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
from .job_type import JobType
from .job_usage import JobUsageStats
from .pydantic_objectid import PydanticObjectId

logger = logging.getLogger(__name__)


class JobIncrementalStatus(FModel):
    status: Optional[str] = None
    start_time: Optional[int | float | datetime] = None
    end_time: Optional[int | float | datetime] = None
    time_now: Optional[int | float | datetime] = None
    progress_counter: Optional[int] = None
    max_progress: Optional[int] = None
    epoch_idx: Optional[int] = None
    epoch_total: Optional[int] = None
    batch_idx: Optional[int] = None
    batch_total: Optional[int] = None
    current_loss: float = 0.0
    validation_loss: Optional[float] = None

    message: Optional[str] = None
    percent_complete: Optional[float] = None


class JobMeta(FeatrixBase):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    """
    This is used to store the data about a job run in the db, but is also used as the meta we exchange with
    the workers (and they with commands/scripts they run) to control and detail the job.
    """

    organization_id: PydanticObjectId
    job_type: JobType
    readonly_object: bool = False

    # These are ids of objects that are involved in the job, if they are at all.  This is
    # different than the changed_ids as those come after, and are what we impacted/changed vs
    # what was used (though there could be overlap).  E.g. A model create will have a project
    # and embedding id.  In fact right now all jobs I think have a project id at least.
    project_id: Optional[PydanticObjectId] = None
    embedding_space_id: Optional[PydanticObjectId] = None
    model_id: Optional[PydanticObjectId] = None
    upload_ids: Optional[List[PydanticObjectId]] = None

    finished: bool = False
    error: bool = False
    message: Optional[str] = None
    error_msg: Optional[str] = None
    error_details: Optional[List[str]] = None
    error_time: Optional[datetime] = None
    exception: Optional[List[str]] = None
    warnings: Optional[List[str]] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    celery_id: Optional[str] = None
    changed_ids: Optional[Dict[str, PydanticObjectId | List[PydanticObjectId]]] = None

    # When the job finishes,
    finished_stats: Optional[JobUsageStats] = None
    # Can be any of the *Args classes in job_requests.py but serialized
    # since pydantic won't know how to serialize/deserialize automatically
    request_args: Optional[Dict] = None
    # request_args: Optional[Dict[str, Any]] = None

    # Incremental stats that occur from the script
    incremental_status: Optional[JobIncrementalStatus] = None

    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # This is the answer that the job comes up with (if any -- some just create things like an embedding_space or model)
    results: Optional[Dict | List] = None

    chained_job_id: Optional[PydanticObjectId] = None

    working_directory: Optional[str] = None
    working_host: Optional[str] = None

    incident: Optional[str | int] = None
    system_meta: Optional[Dict] = None

    # If this job is published from a running job, this will control whether we auto-launch the job as soon as we
    # put it in the database.
    auto_launch: Optional[bool] = None

    # If there is a job that is required to be finished before this job can start, this is the id of that job.
    # This is used for when jobs are created from a Run* script, and there are some dependencies until
    required_job_id: Optional[PydanticObjectId] = None

    # If this job is part of a chain or started as a sub-job as part of a group of jobs, everyone will point back to
    # the start of the chain.  Eventually we will turn this into a dag of jobs and can eliminate some of the
    # waiter jobs we use to stagger things when we don't have chain capabilities (e.g. you start a model separately
    # while the embedding space is still being trained).
    parent_job_id: Optional[PydanticObjectId] = None


class JobInfoResponse(FModel):
    error: bool
    job_id: str
    message: str


class Loss(FModel):
    epoch: int
    current_learning_rate: float
    loss: Any
    validation_loss: float
    time_now: datetime = datetime.utcnow()
    duration: int


class JobStatus(FModel):
    message: str
    finished: bool = False
    error: bool = False
    total: int = 0
    step: int = 0
    percent_complete: float = 0.0
    epoch: int = 0
    epoch_total: int = 0
    error_message: Optional[str] = None
    error_details: Optional[List[str]] = None
    job_result: Optional[Any] = None
    job_meta: Optional[JobMeta] = None  # JobMeta
    status: Optional[Any] = None
    loss_history: Optional[List[Loss]] = None


class JobDispatch(FModel):
    celery_id: Optional[str] = None
    error: bool = False
    job_id: Optional[PydanticObjectId | str] = None
    error_message: Optional[str] = None
    error_detail: Optional[List[str]] = None
