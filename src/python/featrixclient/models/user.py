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
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .featrix_base import FeatrixBase
from .pydantic_objectid import PydanticObjectId

logger = logging.getLogger(__name__)


class User(FeatrixBase):
    first_name: str = ""
    last_name: str
    email: str
    # sms number
    phone_number: Optional[str] = None

    # Supertokens user id
    supertokens_user_id: Optional[str] = None

    # Users can be part of one or more orgs, but can only be active in one
    # at a time (can't see data from multiple orgs at the same time)
    current_organization_id: PydanticObjectId
    # If the user is an admin of an org (for now), we just stuff
    # the org id in this list
    admin_organization_ids: List[PydanticObjectId] = Field(default_factory=list)

    # this should really be links so beanie can pull them
    # in with an aggregation organizations: List[Link["Organization"]]
    organization_ids: List[PydanticObjectId]

    # FIXME: This is set for now on Featrix people who end up admins on all
    #  orgs for support. replace with better RBAC
    super_admin: bool = False

    # This holds the hubspot person ID when we add them to Hubspot (or pull them
    # if they were coming from a sales funnel)
    hubspot_id: Optional[str] = None


class UserBrief(BaseModel):
    id: PydanticObjectId | str = Field(alias="_id")
    first_name: str
    last_name: str
    email: str
    created_at: datetime
    created_by: PydanticObjectId

    model_config = ConfigDict(populate_by_name=True)
