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
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .featrix_base import FeatrixBase
from .pydantic_objectid import PydanticObjectId

logger = logging.getLogger(__name__)

KEYFILE_LOCATION = "~/.featrix.key"


class ApiKey(FeatrixBase):
    """
    This represents a single API key for an organization.  We don't store the secret itself, but a hash of it,
    so if someone gains access to our db somehow, they can't use the contents to run valid operations.

    When a user uses an API key, they provide their own user_id (User.user_id) as the client_id allowing us to
    track WHO in the organization is doing what with the key.

    API Keys can be deactivated, but not deleted -- we preserve them since they tie to auth-ing operations that
    cost money.  This might be overkill, but until we have a fuller system, seems reasonable.

    """

    # The organization to which this key belongs
    organization_id: PydanticObjectId
    # The user that created the key itself -- DEPRECATE FOR created_by
    user_id: PydanticObjectId

    # A hash of the key itself.  We return the actual key to the user and do not store it (so
    # it can't be stolen if we have a breach) -- we use a sha3_512 hash to get something we can
    # use for comparison/validation
    hashed_key: Optional[str] = None

    # The label the user gave us for this key.  This is prepended to the key we return to them
    # for easier management
    label: str


class ApiKeyEntry(BaseModel):
    """
    This is a representation of the API Key entry sans the hashed key for giving back to the user.
    """

    id: PydanticObjectId = Field(alias="_id")

    organization_id: PydanticObjectId
    label: str
    created_at: datetime
    created_by: PydanticObjectId

    model_config = ConfigDict(populate_by_name=True)


class ApiKeyCreated(ApiKeyEntry):
    """
    This is a representation of the API Key entry sans the hashed key for giving back to the user.
    """

    client_secret: str
    client_id: str


class ApiKeyCreateRequest(BaseModel):
    label: str


class ApiKeyAuthenticationRequest(BaseModel):
    client_id: str
    client_secret: str


class ApiKeyAuthResponse(BaseModel):
    jwt: str
    expiration: datetime
