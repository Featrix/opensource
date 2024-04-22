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

from pydantic import Field

from .api_urls import ApiInfo
from .featrix_model import FeatrixModel
from .models import EmbeddingDistanceResponse
from .models import EmbeddingSpace
from .models import PydanticObjectId


class FeatrixEmbeddingSpace(EmbeddingSpace):
    client: Optional[Any] = None
    models_cache: Dict[str, FeatrixModel] = Field(default_factory=dict, exclude=True)
    models_cache_updated: Optional[datetime] = Field(default=None, exclude=True)

    @classmethod
    def all(cls, fc) -> List["FeatrixEmbeddingSpace"]:
        results = fc.api.op("es_get_all")
        return ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=fc)

    @classmethod
    def by_id(cls, fc, es_id: PydanticObjectId | str) -> "FeatrixEmbeddingSpace":
        results = fc.api.op("es_get", embedding_space_id=str(es_id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, results, fc=fc)

    def models(self, force: bool = False):
        since = None
        if self.models_cache_updated is not None and not force:
            since = self.models_cache_updated
            result = self.fc.check_updates(model=self.models_cache_updated)
            if result.model is False:
                return list(self.model_cache.values())
        self.models_cache_updated = datetime.utcnow()
        results = self.fc.api.op(
            "es_get_models", embedding_space_id=str(self._id), since=since
        )
        models = ApiInfo.reclass(FeatrixModel, results, fc=self.fc)
        for model in models:
            # FIXME: The model isn't getting serialized correctly
            if hasattr(model, "_id"):
                model.id = getattr(model, "_id")
            self.models_cache[str(model.id)] = model
        return models

    def model(
        self, model_id: str | PydanticObjectId, force: bool = False
    ) -> FeatrixModel:
        model_id = str(model_id)
        if not force:
            if model_id in self.model_cache:
                return self.model_cache[model_id]
        result = self.fc.api.op(
            "es_get_model", embedding_space_id=str(self.id), model_id=str(model_id)
        )
        model = ApiInfo.reclass(FeatrixModel, result, fc=self.fc)
        self.model_cache[model_id] = model
        return model

    def histogram(self) -> EmbeddingDistanceResponse:
        results = self.fc.api.op("es_get_histogram", embedding_space_id=str(self.id))
        return results

    def distance(self) -> EmbeddingDistanceResponse:
        results = self.fc.api.op("es_get_distance", embedding_space_id=str(self.id))
        return results

    def delete(self) -> "FeatrixEmbeddingSpace":
        result = self.fc.api.op("es_delete", embedding_space_id=str(self.id))
        return ApiInfo.reclass(FeatrixEmbeddingSpace, result, fc=self.fc)
