#  -*- coding: utf-8 -*-
#############################################################################
#
#  Copyright (c) 2024, Featrix, Inc.
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
#############################################################################
#
#     Welcome to...
#
#      _______ _______ _______ _______ ______ _______ ___ ___
#     |    ___|    ___|   _   |_     _|   __ \_     _|   |   |
#     |    ___|    ___|       | |   | |      <_|   |_|-     -|
#     |___|   |_______|___|___| |___| |___|__|_______|___|___|
#
#                                                 Let's embed!
#
#############################################################################
#
#  Sign up for Featrix at https://app.featrix.com/
# 
#############################################################################
#
#  Check out the docs -- you can either call the python built-in help()
#  or fire up your browser:
#
#     https://featrix-docs.readthedocs.io/en/latest/
#
#  You can also join our community Slack:
#
#     https://bits.featrix.com/slack
#
#  We'd love to hear from you: bugs, features, questions -- send them along!
#
#     hello@featrix.ai
#
#############################################################################
#
from __future__ import annotations

import warnings
from collections import namedtuple
from typing import Any

import pydantic
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic import ConfigDict

from .exceptions import FeatrixException
from .models import AllFieldsResponse
from .models import EmbeddingDistanceResponse
from .models import EmbeddingSpace
from .models import EmbeddingSpaceDeleteResponse
from .models import EncodeRecordsArgs
from .models import ESCreateArgs
from .models import Feed
from .models import FeedCreateArgs
from .models import FeedWithEventCounts
from .models import GuardRailsArgs
from .models import JobDispatch
from .models import JobMeta
from .models import JobResults
from .models import Model
from .models import ModelCreateArgs
from .models import ModelFastPredictionArgs
from .models import ModelPredictionArgs
from .models import Organization
from .models import Prediction
from .models import Project
from .models import ProjectAddMappingsRequest
from .models import ProjectAssociateRequest
from .models import ProjectCreateRequest
from .models import ProjectDeleteResponse
from .models import ProjectIgnoreColsRequest
from .models import TrainMoreArgs
from .models import UpdatedResponse
from .models import Upload
from .models import UploadFetchUrlArgs
from .models import User
# from .models import NewNeuralFunctionArgs
# from .models import UserBrief

Api = namedtuple("Api", ["url", "arg_type", "response_type", "list_response"])


class ApiInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    info_get: Api = Api("/info/", None, dict, False)
    users_get_self: Api = Api("/mosaic/users/me/", None, User, False)
    # org_get_all: Api = Api("/mosaic/organizations/", None, Organization, True)
    org_get: Api = Api("/mosaic/organizations/{org_id}", None, Organization, False)
    # activity_get_all: Api = Api("/mosaic/activity/", None, Activity, True)
    # activity_get_by_project: Api = Api(
    #     "/mosaic/activity/project_id/{project_id}", None, Activity, True
    # )
    public_data_get_file: Api = Api("/mosaic/publicdata/", None, FileResponse, False)
    project_get_all: Api = Api("/neural/project/", None, Project, True)
    project_get: Api = Api("/neural/project/{project_id}", None, Project, False)
    project_create: Api = Api("/neural/project/", ProjectCreateRequest, Project, False)
    project_update: Api = Api("/neural/project/edit", Project, Project, False)
    project_get_jobs: Api = Api(
        "/neural/project/{project_id}/jobs", None, JobMeta, True
    )
    project_get_embedding_spaces: Api = Api(
        "/neural/project/{project_id}/embedding_spaces", None, EmbeddingSpace, True
    )
    project_get_models: Api = Api(
        "/neural/project/{project_id}/models", None, Model, True
    )
    project_get_fields: Api = Api(
        "/neural/project/{project_id}/all_fields", None, AllFieldsResponse, True
    )
    project_associate_file: Api = Api(
        "/neural/project/associate", ProjectAssociateRequest, Project, False
    )
    project_add_mapping: Api = Api(
        "/neural/project/mapping", ProjectAddMappingsRequest, Project, False
    )
    project_add_ignore_columns: Api = Api(
        "/neural/project/ignore-cols", ProjectIgnoreColsRequest, Project, False
    )
    project_delete: Api = Api(
        "/neural/project/{project_id}", None, ProjectDeleteResponse, False
    )
    models_get_predictions: Api = Api(
        "/neural/models/{model_id}/predictions", str, Prediction, True
    )
    models_create_prediction: Api = Api(
        "/neural/models/prediction", ModelFastPredictionArgs, Prediction, False
    )
    model_get: Api = Api("/neural/models/{model_id}", None, Model, False)

    uploads_get_all: Api = Api("/neural/data/upload/", None, Upload, True)
    uploads_create: Api = Api("/neural/data/upload/", "files", Upload, False)
    uploads_get: Api = Api("/neural/data/upload/{upload_id}", None, Upload, False)
    uploads_delete: Api = Api("/neural/data/upload/{upload_id}", None, Upload, False)
    uploads_get_by_hash: Api = Api(
        "/neural/data/upload/by_hash/{hash_id}", None, Upload, False
    )
    uploads_get_info: Api = Api(
        "/neural/data/upload/info/{upload_id}", None, Upload, False
    )
    uploads_fetch_url: Api = Api(
        "/neural/data/upload/fetch_url", UploadFetchUrlArgs, Upload, False
    )
    es_get_all: Api = Api("/neural/data/embedding_space/", None, EmbeddingSpace, True)
    es_get: Api = Api(
        "/neural/embedding_space/{embedding_space_id}", None, EmbeddingSpace, False
    )
    es_get_training_jobs: Api = Api(
        "/neural/embedding_space/{embedding_space_id}/training_jobs",
        None,
        JobMeta,
        True,
    )
    es_get_model: Api = Api(
        "/neural/embedding_space/{embedding_space_id}/model/{model_id}",
        None,
        Model,
        False,
    )
    es_get_models: Api = Api(
        "/neural/embedding_space/{embedding_space_id}/models",
        None,
        Model,
        True,
    )
    es_get_histogram: Api = Api(
        "/neural/embedding_space/{embedding_space_id}/value-historgram",
        None,
        None,
        False,
    )
    es_get_distance: Api = Api(
        "/neural/embedding_space/{embedding_space_id}/embedding-distance",
        None,
        EmbeddingDistanceResponse,
        False,
    )
    es_delete: Api = Api(
        "/neural/embedding_space/{embedding_space_id}",
        None,
        EmbeddingSpaceDeleteResponse,
        False,
    )
    # es_get_explorer: Api = Api(
    #     "/neural/embedding_space/{embedding_space_id}/explorer", None, dict, False
    # )
    feeds_get: Api = Api("/neural/feeds/", None, FeedWithEventCounts, True)
    feeds_get_create_feed: Api = Api(
        "/neural/feeds/create", FeedCreateArgs, Feed, False
    )
    feeds_create_event: Api = Api("/neural/feeds/{feed_public_id}", None, str, False)
    # Create ES will produce a create-embedding and create-projection
    job_es_create: Api = Api(
        "/neural/embedding_space/", ESCreateArgs, JobDispatch, True
    )
    job_es_train_more: Api = Api(
        "/neural/embedding_space/trainmore", TrainMoreArgs, JobDispatch, False
    )
    job_model_create: Api = Api(
        "/neural/embedding_space/train-downstream-model",
        ModelCreateArgs,
        JobDispatch,
        False,
    )
    job_model_prediction: Api = Api(
        "/neural/embedding_space/run-model-prediction",
        ModelPredictionArgs,
        JobDispatch,
        False,
    )
    job_model_guardrails: Api = Api(
        "/neural/embedding_space/check-guardrails", GuardRailsArgs, JobDispatch, False
    )
    job_fast_encode_records: Api = Api(
        "/neural/embedding_space/fast-encode-records",
        EncodeRecordsArgs,
        Any,
        False,  # this only matters if the response type is a Model (pydantic) - here we just get a list of dicts
    )
    # job_es_create_db: Api = Api(
    #     "/neural/embedding_space/create-database", CreateDBArgs, JobDispatch, False
    # )
    # job_db_cluster: Api = Api(
    #     "/neural/embedding_space/run-db-cluster", DBClusterArgs, JobDispatch, False
    # )
    # job_db_nn_query: Api = Api(
    #     "/neural/embedding_space/nn-query", NNQueryArgs, JobDispatch, False
    # )
    # job_chained_new_neural_function: Api = Api(
    #     "/neural/project/new-neural-function", NewNeuralFunctionArgs, JobDispatch, True
    # )
    # job_chained_new_explorer: Api = Api(
    #     "/neural/project/new-explorer", NewExplorerArgs, JobDispatch, True
    # )
    jobs_get: Api = Api("/neural/job/{job_id}", None, JobResults, False)

    @staticmethod
    def verb(name: str):
        if "get" in name:
            return "get"
        if name.startswith("job"):
            return "job"
        # In the long term we probably want post/put depending on create vs update (and to return 201/200 for created
        # vs ok ? But for now it's all just a post
        for _op in ["post", "update", "create", "associate"]:
            if _op in name:
                return "post"
        if "delete" in name:
            return "delete"
        raise FeatrixException(f"Unknown verb for operation {name}")

    def get(self, name: str) -> Api | None:
        return getattr(self, name, None)

    @staticmethod
    def featrix_validate(api_name, response_object):
        api = ApiInfo().get(api_name)
        if api.response_type is None:
            return None
        if api.response_type == Any:
            return response_object
        
        if issubclass(api.response_type, BaseModel):
            try:
                if api.list_response:
                    ro_list = [
                        _ro.model_dump() if isinstance(_ro, BaseModel) else _ro
                        for _ro in response_object
                    ]
                    return [api.response_type.model_validate(_) for _ in ro_list]
                else:
                    ro = (
                        response_object.model_dump()
                        if isinstance(response_object, BaseModel)
                        else response_object
                    )
                    return api.response_type.model_validate(ro)
            except pydantic.ValidationError as e:
                print("@@@ response_object = ", response_object)
                warnings.warn(f"Invalid response: it is possible you need a newer client: {e}")
                # ApiInfo.dump_validation_error_details(
                #     e, response_object, api.response_type
                # )
                return response_object
        if not isinstance(response_object, api.response_type):
            return api.response_type(response_object)  # noqa -- will be a base type like int, float, etc
        return response_object

    @staticmethod
    def reclass(parent, model, fc: Any = None, **kwargs):
        def augment(obj):
            if fc:
                setattr(obj, "_fc", fc)
            for k, v in kwargs.items():
                setattr(obj, k, v)
            # For now, fix the issue with _id / id
            ident_id = getattr(obj, "_id", None)
            if ident_id is not None:
                setattr(obj, "id", ident_id)
            return obj

        if not issubclass(parent, BaseModel):
            raise ValueError("Cannot reclass non-pydantic classes")
        if isinstance(model, list):
            return [augment(parent.model_validate(_.model_dump())) for _ in model]
        # print(f"Passing to validate for {parent}: {model.model_dump_json(indent=4)}")
        return augment(parent.model_validate(model.model_dump()))

    @staticmethod
    def url_substitution(url, **kwargs):
        return url.format(**kwargs)
