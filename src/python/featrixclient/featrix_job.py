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

import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import PrivateAttr

from .api_urls import ApiInfo
from .exceptions import FeatrixException
from .models import GuardRailsArgs
from .models import PydanticObjectId
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
    Represents a job running on a Featrix cluster.

    Key Fields:
    -----------
    - `job.finished` : bool
        True if the job is finished.
    - `job.error` : bool
        True if the job encountered an error.
    - `job.incremental_status` : JobStatus
        Current status, including `job.incremental_status.message` for progress updates.

    Methods:
    --------
    - `by_id(job_id, fc)` : FeatrixJob
        Retrieve a job by ID using a `FeatrixClient` instance.
    - `refresh()` : FeatrixJob
        Update the job with the latest status from the server.
    - `wait_for_completion()` : FeatrixJob
        Blocks until the job completes, printing status updates.

    Related Methods:
    ----------------
    - `by_project`, `by_embedding_space`, `by_model`, `by_upload`
        Retrieve jobs related to specific objects like projects, embedding spaces, models, and uploads.
    """

    _fc: Optional[Any] = PrivateAttr(default=None)
    """
    Reference to the Featrix class that retrieved or created this project, used for API calls/credentials
    """
    # _latest_job_result: Optional[JobResults] = PrivateAttr(default=None)

    @classmethod
    def by_id(
        cls, job_id: str | PydanticObjectId, fc: Optional["Featrix"] = None
    ) -> "FeatrixJob":  # noqa F821
        """
        Retrieve a job by its job ID from the server

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
        job = ApiInfo.reclass(cls, results.job_meta, fc=fc)
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
    def by_neural_function(cls, model: "FeatrixNeuralFunction") -> List["FeatrixJob"]:  # noqa F821 forward ref
        from .featrix_neural_function import FeatrixNeuralFunction  # noqa F821

        project = model.fc.get_project_by_id(model.project_id)
        jobs = project.jobs()
        model_jobs = []
        for job in jobs:
            if str(job.model_id) == str(model.id):
                model_jobs.append(job)
        return model_jobs

    @classmethod
    def by_embedding_space(
        cls, embedding_space: "FeatrixEmbeddingSpace"
    ) -> List["FeatrixJob"]:  # noqa F821 forward ref
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
        cycle: int = 5,
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
                    full_msg += (
                        f"\n ...Running Job {job.id}: {job.incremental_status.message}"
                    )
            display_message(full_msg)
            time.sleep(cycle)
        return

    def wait_for_completion(self, message: Optional[str] = None) -> "FeatrixJob":  # noqa
        if self.finished:
            return self
        job = self
        print(f"waiting for completion of job {job.id}")
        while job.finished is False:
            display_message(
                f"{message if message else 'Status:'} "
                f"{job.incremental_status.message if job.incremental_status else 'No status yet'}"
            )
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
        if jd.error:
            raise FeatrixException(jd.error_message)
        return FeatrixJob.by_id(str(jd.job_id), fc)
