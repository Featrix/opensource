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

from typing import Optional

from .fmodel import FModel
from .pydantic_objectid import PydanticObjectId
from .upload import Upload


class UploadAssociation(FModel):
    """
    Associates an uploaded_file CSV file in the customers library with this Project and targets
    a specific column in the CSV and a target Sample % or Row Count.

    If the customer adds an association with a CSV the project already has, the new one
    replaces the old one.
    """

    # Where to find the CSV in the Upload collection
    upload_id: PydanticObjectId
    # customers name of association
    label: str
    sample_percentage: float = 0.0
    sample_row_count: int = 0.0
    drop_duplicates: bool = False

    # The save() will exclude this from the db -- we should be using links here but ... not yet.  FIXME: links
    upload: Optional[Upload] = None
