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

import copy
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from .pydantic_objectid import PydanticObjectId
# from pydantic.fields import FieldInfo
# from pydantic_core import ArgsKwargs
# class NoValidator:
#     def validate_python(self, args_kwargs: ArgsKwargs, self_instance):
#         print("validate_python called")
#         model_fields = cast(dict[str, FieldInfo], self_instance.__pydantic_fields__)
#         # print(model_fields)
#         args = list(args_kwargs.args)
#         mapping = dict(args_kwargs.kwargs) if args_kwargs.kwargs else dict()
#         for field, info in model_fields.items():
#             assert not info.init_var
#             if not info.kw_only:
#                 mapping[field] = args.pop(0)
#         for field, value in mapping.items():
#             setattr(self_instance, field, value)

class FeatrixBase(BaseModel):
    id: PydanticObjectId = Field(default_factory=PydanticObjectId)
    created_by: Optional[PydanticObjectId | str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # @classmethod
    # def unvalidated(__pydantic_cls__: "Type[Model]", **data: Any) -> "Model":
    #     for name, field in __pydantic_cls__.__fields__.items():
    #         try:
    #             data[name]
    #         except KeyError:
    #             if field.required:
    #                 raise TypeError(f"Missing required keyword argument {name!r}")
    #             if field.default is None:
    #                 # deepcopy is quite slow on None
    #                 value = None
    #             else:
    #                 value = copy.deepcopy(field.default)
    #             data[name] = value
    #     self = __pydantic_cls__.__new__(__pydantic_cls__)
    #     object.__setattr__(self, "__dict__", data)
    #     object.__setattr__(self, "__fields_set__", set(data.keys()))
    #     return self

# FeatrixBase.__pydantic_validator__ = NoValidator()
