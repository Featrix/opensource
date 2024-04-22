#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Featrix, Inc, All Rights Reserved
#
#  Proprietary and Confidential.  Unauthorized use, copying or dissemination
#  of these materials is strictly prohibited.
#
from __future__ import annotations

import logging
from enum import Enum
from typing import Dict
from typing import Optional

from pydantic import Field

from .featrix_base import FeatrixBase
from .pydantic_objectid import PydanticObjectId

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    PROJECT_CREATED = "project.created"
    PROJECT_DELETED = "project.deleted"
    PROJECT_RENAMED = "project.renamed"
    PROJECT_DATA_ASSOCIATED = "project.data.associated"
    PROJECT_DATA_UNASSOCIATED = "project.data.unassociated"

    NF_CREATE = "neural-function.created"
    NF_DELETE = "neural-function.deleted"
    NF_TRAINED = "neural-function.trained"

    ES_CREATE = "embedding-space.created"

    DATA_UPLOADED = "data.uploaded"
    DATA_UPLOAD_DELETED = "data.deleted"

    # ES_WAITING_FOR_TRAINING = "embedding-space.waiting"

    API_KEY_CREATED = "api.key.created"
    API_KEY_DELETED = "api.key.deleted"

    ORG_CREATED = "org.created"
    ORG_RENAMED = "org.renamed"
    ORG_DELETED = "org.deleted"

    INVITE_SENT = "org.invite_sent"
    INVITE_ACCEPTED = "org.invite_accepted"

    USER_CREATED = "user.created"
    USER_DELETED = "user.deleted"

    EMBEDDING_CREATED = "embedding.created"
    EMBEDDING_DELETED = "embedding.deleted"

    FEED_CREATED = "feed.created"
    FEED_DELETED = "feed.deleted"

    EMAIL_VERIFIED = "email.verified"
    # ... etc.


class Activity(FeatrixBase):
    """
    This is an activty record:

        - project changes
            - data added
            - mapping changed
            -
        - org changes
            - billing charged / changed
            - org name
            -
        - user changes
            - name
            - password
            - priv level
            - api key
        - invitation created at
        - neural function created
        -   trained
        -   more trained
        -   "foundation" trained
        -   "function" trained
        - prediction roll up [say 1 hr of stuff gets turned into 1 event]
        - file uploaded
        - daily event of past day's usage?
        -

    #user_id --> created_by
    #when_time --> created_at
    org_id
    project_id  # can be null

    """

    # The organization to which this key belongs
    organization_id: PydanticObjectId
    impacted_project_id: Optional[PydanticObjectId] = None
    impacted_user_id: Optional[PydanticObjectId] = None
    impacted_model_id: Optional[PydanticObjectId] = None
    impacted_embedding_space_id: Optional[PydanticObjectId] = None
    impacted_upload_id: Optional[PydanticObjectId] = None
    impacted_api_key_id: Optional[PydanticObjectId] = None
    impacted_feed_id: Optional[PydanticObjectId] = None

    # impacted_billing_id: Optional[PydanticObjectId] = None

    activity_type: ActivityType

    # FIXME: this doesn't work - somehow it confuses mongo find and we get an error on "id" being a dict not objectid??
    # details: Dict[str, PydanticObjectId | Any] = Field(default_factory=dict)
    details: Dict = Field(default_factory=dict)
