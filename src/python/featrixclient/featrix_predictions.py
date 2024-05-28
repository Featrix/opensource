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

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import Field

from .api_urls import ApiInfo
from .models import Prediction


class FeatrixPrediction(Prediction):
    """
    Represents a prediction that was run against a neural function (aka predictive model).
    """
    fc: Optional[Any] = Field(default=None, exclude=True)
    """Reference to the Featrix class  that retrieved or created this project, used for API calls/credentials"""


    @classmethod
    def all(
        cls,
        fc: Any,
        model: Optional["FeatrixModel" | str] = None,  # noqa F821
    ) -> List[FeatrixPrediction]:
        """
        Get all predictions for a given model.

        Args:
            fc: Featrix class instance
            model: Model instance or model_id

        Returns:
            List[FeatrixPrediction]: List of predictions
        """
        from .featrix_model import FeatrixModel

        model_id = model.id if isinstance(model, FeatrixModel) else model
        predictions = fc.api.op("models_get_predictions", model_id=model_id)
        return ApiInfo.reclass(cls, predictions, fc=fc)

    def match(self, **kwargs) -> Tuple[bool, int]:
        """
        Check the list of queries in this prediction to see if it matches the query provided in kwargs.

        Args:
            **kwargs: Query to match (key=value pairs for a single query)

        Returns:
            Tuple[bool, int]: True if the query matches, and the index of the query in the list of queries
        """
        ks = set(kwargs.keys())
        for idx, query in enumerate(self.query):
            ps = set(query.keys())
            if len(ks) != len(ps):
                return False, -1
            if len(ks - ps) > 0:
                return False, -1
            for query_col, value in kwargs.items():
                if value != query.get(query_col):
                    break
            else:
                return True, idx
        return False, -1
