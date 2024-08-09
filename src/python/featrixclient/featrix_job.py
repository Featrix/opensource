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
import warnings
import time

from IPython.display import clear_output
from pydantic import Field, PrivateAttr

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .models import ESCreateArgs, PydanticObjectId
from .models import GuardRailsArgs
from .models import JobResults
from .models import ModelPredictionArgs
from .models.job_meta import JobDispatch
from .models.job_meta import JobMeta as Job
from .utils import display_message


#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2024 Featrix, Inc, All Rights Reserved
#
#  Proprietary and Confidential.  Unauthorized use, copying or dissemination
#  of these materials is strictly prohibited.


class FeatrixJob(Job):
    """
    This represents a job that the Featrix server is running on one of its neural compute clusters on your behalf.
    It contains full details about the job, including, if it is still running, the incremental status of the job
    itself.

    A job has a few key fields that indicate their state:
        job.finished: bool: True if the job is finished
        job.error: bool: True if the (finished) job had an error
        job.incremental_status: JobStatus: the incremental status of the job, if it is still running -- the field
                    job.incremental_status.message will have a human-readable message about the job's progress

    If you have the job id, you can get it by using the .by_id method, which will return a new instance of the job
    specified.  If you already have a job, you can use job.refresh() to pull any updates from the server for that job:
        job = FeatrixJob.by_id(job_id, fc)  # pass in your FeatrixClient instance
        job = job.refresh()

    There are a handful of class methods for getting jobs that are related to other objects, such as
    projects (by_project), embedding spaces (by_embedding_space), models (by_model), and uploads (by_upload).

    If you have a job that is currently running, you can also call job. wait_for_completion() to block until the job
    is finished (qne print out any updates to the job.incremental_status.message field).  This will return the
    current version (updated) of the Job in a new object:
            job = job.wait_for_completion("Waiting for job to complete: ")
            assert job.finished is True

    """
    _fc: Optional[Any] = PrivateAttr(default=None)
    """Reference to the Featrix class  that retrieved or created this project, used for API calls/credentials"""
    # _latest_job_result: Optional[JobResults] = PrivateAttr(default=None)

    @classmethod
    def by_id(cls, job_id: str | PydanticObjectId, fc: Optional["Featrix"] = None) -> "FeatrixJob":  # noqa F821
        """
        REtrieve a job by its job ID from the server, ignoring cache

        Arguments:
            job_id: str: the job ID to retrieve
            fc: Featrix: the Featrix class that is making the request

        Returns:
            FeatrixJob new instance of job
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()

        results = fc.api.op("jobs_get", job_id=str(job_id))
        # print(f"Got {results} from job get")
        # print(f"job meta is {results.job_meta}")
        job = ApiInfo.reclass(cls, results.job_meta, fc=fc)
        # job._latest_job_results = results
        return job

    @property
    def fc(self):
        return self._fc

    @fc.setter
    def fc(self, value):
        from .networkclient import Featrix

        if isinstance(value, Featrix) is False:
            raise FeatrixException("fc must be an instance of Featrix")
        self._fc = value

    def refresh(self):
        return self.by_id(self.id, self.fc)

    # def job_results(self):
    #     """
    #     Return the lateest job results for this job.  This will be None if the job is still running.
    #     """
    #     return self._latest_job_result

    # def message(self):
    #     """
    #
    #     """
    #     return self._latest_job_result.message if self._latest_job_result else None

    def check(self):
        return FeatrixJob.by_id(str(self.id), self._fc)

    @classmethod
    def by_model(cls, model: "FeatrixNeuralFunction") -> List["FeatrixJob"]:  # noqa F821 forward ref
        from .featrix_neural_function import FeatrixNeuralFunction  # noqa F821

        project = model.fc.get_project_by_id(model.project_id)
        jobs = project.jobs()
        model_jobs = []
        for job in jobs:
            if str(job.model_id) == str(model.id):
                model_jobs.append(job)
        return model_jobs

    by_neural_function = by_model

    @classmethod
    def by_embedding_space(cls, embedding_space: FeatrixEmbeddingSpace) -> List["FeatrixJob"]:  # noqa F821 forward ref
        project = embedding_space.fc.get_project_by_id(embedding_space.project_id)

        jobs = project.jobs()
        es_jobs = []
        for job in jobs:
            if str(job.embedding_space_id) == str(embedding_space.id):
                es_jobs.append(job)
        return es_jobs

    @classmethod
    def by_project(cls, project: "FeatrixProject") -> List["FeatrixJob"]:  # noqa F821 forward ref
        from .featrix_project import FeatrixProject  # noqa F821

        return project.jobs()

    @classmethod
    def by_upload(cls, upload: "FeatrixUpload") -> List["FeatrixJob"]:  # noqa F821 forward ref
        from .featrix_upload import FeatrixUpload  # noqa F821

        # we return a list here to be consistent with the other by_* methods
        return [cls.by_id(upload.post_processing_job_id, upload.fc)]

    @staticmethod
    def wait_for_jobs(
            fc: Any,
            jobs: List["FeatrixJob"],
            msg: str = "Waiting for jobs: ",
            cycle: int = 5
    ) -> List[FeatrixJob]:
        """
        Given a list of FeatrixJob's, wait for all of them to be completed, periodically updating their status on
        the console.

        Arguments:
            fc: Featrix: the Featrix class that is making the request
            jobs: List[FeatrixJob]: the list of jobs to wait for
            msg: str: the message to display on the console
            cycle: int: the number of seconds to wait between updates

        Returns:
            List[FeatrixJob]: the updated list of jobs that have been completed
        """
        cnt = len(jobs)
        while True:
            done = errors = 0
            jobs = [job.by_id(str(job.id), fc) for job in jobs]
            working = []
            for job in jobs:
                if job.finished is True:
                    done += 1
                    if job.error is True:
                        errors += 1
                elif job.incremental_status and job.incremental_status.message:
                    working.append(job)
            if cnt == done:
                return jobs
            errmsg = f" -- errors {errors}" if errors else ""
            full_msg = f"{msg}: {done}/{cnt} completed {errmsg}"
            if len(working):
                for job in working:
                    full_msg += f"\n ...Running Job {job.id}: {job.incremental_status.message}"
            display_message(full_msg)
            time.sleep(cycle)

    def wait_for_completion(self, message: Optional[str] = None) -> "FeatrixJob":  # noqa
        if self.finished:
            return self
        job = self
        print(f"waiting for completion of job {job.id}")
        while job.finished is False:
            display_message(f"{message if message else 'Status:'} "
                            f"{job.incremental_status.message if job.incremental_status else 'No status yet'}")
            time.sleep(5)
            job = job.by_id(str(self.id), self._fc)
        display_message(
            f"{message if message else 'Status:'} "
            f"{job.incremental_status.message if job.incremental_status else 'Completed'}"
        )
        return job

    @classmethod
    def check_guardrails(
        cls,
        fc: Any,
        model: "FeatrixNeuralFunction",  # noqa F821 forward ref
        query: List[Dict] | Dict,
        issues_only: bool = False,
    ) -> "FeatrixJob":
        """
        Kick off a Check Guardrails on a query or list of queries.

        Arguments:
            fc: Featrix: the Featrix class that is making the request
            model: FeatrixNeuralFunction: the model to check guardrails on
            query: List[Dict] | Dict: the query or list of queries to check
            issues_only: bool: if True, only return the issues, not the full results

        Returns:
            FeatrixJob: the job that is running the guardrails check
        """
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
        if jd.error:
            raise FeatrixException(jd.error_message)
        return FeatrixJob.by_id(str(jd.job_id), fc)


