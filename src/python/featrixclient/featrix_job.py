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

from pydantic import Field

from .api_urls import ApiInfo
from .featrix_model import FeatrixModel
from .models import ESCreateArgs
from .models import GuardRailsArgs
from .models import JobResults
from .models import ModelPredictionArgs
from .models.job_meta import JobDispatch
from .models.job_meta import JobMeta as Job


#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Featrix, Inc, All Rights Reserved
#
#  Proprietary and Confidential.  Unauthorized use, copying or dissemination
#  of these materials is strictly prohibited.


class FeatrixJob(Job):
    fc: Optional[Any] = Field(default=None, exclude=True)
    latest_job_result: Optional[JobResults] = None

    @classmethod
    def by_id(cls, fc: Any, job_id: str) -> "FeatrixJob":
        results = fc.api.op("jobs_get", job_id=job_id)
        # print(f"Got {results} from job get")
        # print(f"job meta is {results.job_meta}")
        job = ApiInfo.reclass(cls, results.job_meta, fc=fc)
        job.job_results = results
        return job

    def job_results(self):
        return self.latest_job_result

    def message(self):
        return self.latest_job_result.message if self.latest_job_result else None

    def check(self):
        return FeatrixJob.by_id(self.fc, str(self.id))

    @classmethod
    def create_embedding(
        cls,
        fc,
        project,  # FeatrixProject
        **kwargs,
    ):
        es_create = ESCreateArgs(project_id=str(project.id), **kwargs)
        result = fc.api.op("jobs_es_create", es_create)
        job = cls.from_job_dispatch(result, fc)
        return job

    @classmethod
    def run_prediction(
        cls,
        fc: Any,
        model: FeatrixModel,
        query: List[Dict] | Dict,
    ):
        if isinstance(query, dict):
            query = [query]
        predict = ModelPredictionArgs(model_id=str(model.id), query=query)
        result = fc.api.op("job_model_prediction", predict)
        job = cls.from_job_dispatch(result, fc)
        return job

    @classmethod
    def check_guardrails(
        cls,
        fc: Any,
        model: FeatrixModel,
        query: List[Dict] | Dict,
        issues_only: bool = False,
    ):
        check = GuardRailsArgs(
            model_id=str(model.id), issues_only=issues_only, query=query
        )
        result = fc.api.op("jobs_predict", check)
        job = cls.from_job_dispatch(result, fc)
        return job

    @classmethod
    def from_job_dispatch(cls, jd: JobDispatch, fc) -> "FeatrixJob":
        # print(f"Job from dispatch is {str(jd.job_id)}")
        # print(f"Job dispatch is {jd.model_dump_json(indent=4)}")
        return FeatrixJob.by_id(fc, str(jd.job_id))


"""
    job_es_create: Api = Api('/neural/embedding_space/', ESCreateArgs, JobDispatch, False)
    job_es_train_more: Api = Api('/neural/embedding_space/trainmore', TrainMoreArgs, JobDispatch, False)
    job_model_create: Api = Api('/neural/embedding_space/train-downstream-model', ModelCreateArgs, JobDispatch, False)
    job_model_prediction: Api = Api('/neural/embedding_space/run-model-prediction',
                                    ModelPredictionArgs, JobDispatch, False)
    job_model_guardrails: Api = Api('/neural/embedding_space/check-guardrails', GuardRailsArgs, JobDispatch, False)
    job_encode_records: Api = Api('/neural/embedding_space/run-encode-records', EncodeRecordsArgs, JobDispatch, False)
    job_es_create_db: Api = Api('/neural/embedding_space/create-database', CreateDBArgs, JobDispatch, False)
    job_db_cluster: Api = Api('/neural/embedding_space/run-db-cluster', DBClusterArgs, JobDispatch, False)
    job_db_nn_query: Api = Api('/neural/embedding_space/nn-query', NNQueryArgs, JobDispatch, False)
    job_chained_new_neural_function: Api = Api('/neural/project/new-neural-function',
                                               NewNeuralFunctionArgs, JobDispatch, False)
                                               
                                               """
