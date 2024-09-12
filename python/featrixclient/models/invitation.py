# #  -*- coding: utf-8 -*-
# #############################################################################
# #
# #  Copyright (c) 2024, Featrix, Inc. All rights reserved.
# #
# #  Permission is hereby granted, free of charge, to any person obtaining a copy
# #  of this software and associated documentation files (the "Software"), to deal
# #  in the Software without restriction, including without limitation the rights
# #  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# #  copies of the Software, and to permit persons to whom the Software is
# #  furnished to do so, subject to the following conditions:
# #
# #  The above copyright notice and this permission notice shall be included in all
# #  copies or substantial portions of the Software.
# #
# #  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# #  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# #  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# #  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# #  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# #  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# #  SOFTWARE.
# #
# #
# #############################################################################
# #
# #  Yes, you can see this file, but Featrix, Inc. retains all rights.
# #
# #############################################################################
# from __future__ import annotations
# from datetime import datetime
# from enum import Enum
# from typing import Optional
# from pydantic import BaseModel
# from pydantic import EmailStr
# from pydantic import Field
# from .featrix_base import FeatrixBase
# from .pydantic_objectid import PydanticObjectId
# class InviteState(Enum):
#     invited = "invited"
#     accepted = "accepted"
# class Invitation(FeatrixBase):
#     invited_user_email: EmailStr
#     invite_email_sent_at: datetime = Field(default_factory=datetime.utcnow)
#     invite_accepted_at: datetime | None = None
#     invite_state: InviteState = InviteState.invited
#     # Make the user an admin when they accept the initiation
#     admin_invite: bool = False
#     invited_by: PydanticObjectId
#     organization_id: PydanticObjectId
# class InviteUserRequest(BaseModel):
#     email: EmailStr
#     admin_access: bool
# class InviteUserResponse(BaseModel):
#     status: InviteState
#     email_sent: bool
#     email_error: Optional[str] = None
# class IsInvitedResponse(BaseModel):
#     invited: bool
#     invitation: Invitation | None
# class InvitationBrief(BaseModel):
#     invited_user_email: EmailStr
#     invite_email_sent_at: datetime = Field(default_factory=datetime.utcnow)
#     invite_accepted_at: datetime | None = None
#     invite_state: InviteState = InviteState.invited
#     # Make the user an admin when they accept the initiation
#     admin_invite: bool = False
#     invited_by: PydanticObjectId
#     organization_id: PydanticObjectId
