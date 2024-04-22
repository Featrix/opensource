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

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import Field

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .featrix_predictions import FeatrixPrediction
from .models import Model
from .models import ModelFastPredictionArgs


class FeatrixModel(Model):
    fc: Optional[Any] = Field(default=None, exclude=True)
    predictions_cache: Optional[Any] = Field(default_factory=dict, exclude=True)
    predictions_cache_updated: Optional[datetime] = Field(default=None, exclude=True)

    def predict(self, query: Dict | List[Dict], wait_for_job: bool = True):
        if isinstance(query, dict):
            query = [query]
        predict_args = ModelFastPredictionArgs(model_id=str(self.id), query=query)
        pred = self.fc.api.op("models_create_prediction", predict_args)
        fp = ApiInfo.reclass(FeatrixPrediction, pred, fc=self.fc)
        self.predictions_cache[fp.id] = fp
        if self.fc.debug:
            print(f"Prediction {fp.id}: {fp.query} -> {fp.result}")
            print(
                f"Prediction timing: Overall {fp.debug_info.get('api_time', -1)} "
                f"Query {fp.debug_info.get('prediction_time', -1)}"
            )
        return fp.result

    def predictions(self, force: bool = False):
        from .featrix_predictions import FeatrixPrediction

        since = None
        if self.predictions_cache_updated is not None and not force:
            since = self.predictions_cache_updated
            result = self.fc.check_updates(job_meta=self.predictions_cache_updated)
            if result.job_meta is False:
                return list(self.predictions_cache.values())
        self.predictions_cache_updated = datetime.utcnow()
        results = self.fc.api.op(
            "models_get_predictions", model_id=str(self.id), since=since
        )
        prediction_list = ApiInfo.reclass(FeatrixPrediction, results, fc=self.fc)
        for prediction in prediction_list:
            self.predictions_cache[str(prediction.id)] = prediction
        return prediction_list

    def get_prediction(self, prediction_id: str) -> FeatrixPrediction:
        return self.predictions_cache.get(str(prediction_id))

    def find_prediction(self, **kwargs) -> Tuple[FeatrixPrediction, Dict]:
        """
        Given a set of column to value's that make up a query, look to see if that query is contained in
        any of our predictions -- and if so, return it. Otherwise, return None.
        """
        for prediction in self.predictions_cache.values():
            match, idx = prediction.match(**kwargs)
            if match:
                return prediction, prediction.result[idx]
        raise FeatrixException("No prediction found for query")
