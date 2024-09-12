#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Featrix, Inc, All Rights Reserved
#
#  Proprietary and Confidential.  Unauthorized use, copying or dissemination
#  of these materials is strictly prohibited.
#
from __future__ import annotations

import logging
import uuid
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .featrix_base import FeatrixBase
from .fmodel import FModel
from .pydantic_objectid import PydanticObjectId

logger = logging.getLogger(__name__)


class Feed(FeatrixBase):
    """
    This is an feed record:

        - org id
        - feed name
        - cors_policy ... not sure what this means.
        - feed_public_id -- the id used in public for posting to.

    """

    # The organization to which this key belongs
    organization_id: PydanticObjectId
    feed_name: str
    post_policy: Optional[
        List[Dict]
    ]  # FIXME: I'm not quite sure what goes in here -- cross-origin stuff, maybe other things.
    feed_public_id: Optional[str] = None  # str(uuid.uuid4())

    @classmethod
    def new(
        cls,
        feed_name: str,
        session: Optional[FModel] = None,
        user: Optional[FModel] = None,
        user_id: Optional[PydanticObjectId] = None,
        organization: Optional[FModel] = None,
        organization_id: Optional[PydanticObjectId] = None,
        **kwargs,
    ) -> Feed:
        feed_public_id = str(uuid.uuid4())

        # Do we need a prefix for local?
        created_by = cls.get_id(user_id, user, session.user if session else None)
        organization_id = cls.get_id(
            organization_id, organization, session.organization if session else None
        )

        feed = cls(
            created_by=created_by,
            organization_id=organization_id,
            feed_name=feed_name,
            post_policy=[],
            feed_public_id=feed_public_id,
        )

        return feed


class FeedWithEventCounts(Feed):
    number_events: Optional[int] = 0


class FeedCreateArgs(FModel):
    feed_name: str


class FeedEventEntry(FeatrixBase):
    organization_id: PydanticObjectId
    feed_id: PydanticObjectId
    feed_public_id: str
    data: Any
