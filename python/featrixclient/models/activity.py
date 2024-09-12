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

    EXPLORER_CREATE = "explorer.created"
    EXPLORER_EXTENDED = "explorer.extended"

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
