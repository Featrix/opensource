# -*- coding: utf-8 -*-
#
#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Featrix, Inc, All Rights Reserved
#
#  Proprietary and Confidential.  Unauthorized use, copying or dissemination
#  of these materials is strictly prohibited.
#
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel
from pydantic import EmailStr
from pydantic import Field

from .featrix_base import FeatrixBase
from .pydantic_objectid import PydanticObjectId


class InviteState(Enum):
    invited = "invited"
    accepted = "accepted"


class Invitation(FeatrixBase):
    invited_user_email: EmailStr
    invite_email_sent_at: datetime = Field(default_factory=datetime.utcnow)
    invite_accepted_at: datetime | None = None
    invite_state: InviteState = InviteState.invited
    # Make the user an admin when they accept the initiation
    admin_invite: bool = False
    invited_by: PydanticObjectId
    organization_id: PydanticObjectId


class InviteUserRequest(BaseModel):
    email: EmailStr
    admin_access: bool


class InviteUserResponse(BaseModel):
    status: InviteState
    email_sent: bool
    email_error: Optional[str] = None


class IsInvitedResponse(BaseModel):
    invited: bool
    invitation: Invitation | None


class InvitationBrief(BaseModel):
    invited_user_email: EmailStr
    invite_email_sent_at: datetime = Field(default_factory=datetime.utcnow)
    invite_accepted_at: datetime | None = None
    invite_state: InviteState = InviteState.invited
    # Make the user an admin when they accept the initiation
    admin_invite: bool = False
    invited_by: PydanticObjectId
    organization_id: PydanticObjectId
