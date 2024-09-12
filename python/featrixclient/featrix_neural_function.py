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

import uuid
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import bson
import pandas as pd
from pydantic import PrivateAttr

from .api_urls import ApiInfo
from .config import settings
from .exceptions import FeatrixException
from .featrix_job import FeatrixJob
from .featrix_predictions import FeatrixPrediction
from .models import JobType
from .models import Model
from .models import ModelCreateArgs
from .models import ModelFastPredictionArgs
from .models import PydanticObjectId
from .models import TrainingState


class FeatrixNeuralFunction(Model):
    """
    Represents a neural function (predictive model) trained against an embedding space for predicting a specific feature or field.

    Create and train a new neural function with `.new_neural_function()`. Retrieve an existing neural function by its ID with `.by_id()`, and ensure you have the latest version of the model using `.refresh()`:

        nf = FeatrixNeuralFunction.by_id("5f7b1b1b1b1b1b1b1b1b1b")
        nf = nf.refresh()  # Update after training completes

    """

    _fc: Optional[Any] = PrivateAttr(default=None)
    """Reference to the Featrix class  that retrieved or created this project, used for API calls/credentials"""

    @staticmethod
    def create_args(
        project_id: str, 
        embedding_space_id: str,
        target_field: str, 
        target_field_type: str,
        encoder,
        **kwargs) -> ModelCreateArgs:
        """
        Create the arguments for creating a model.  This is a helper function to make it easier to create a
        `ModelCreateArgs`
        """
        model_create_args = ModelCreateArgs(
            project_id=project_id,
            embedding_space_id=embedding_space_id,
            target_columns=[target_field],
            target_field_type=target_field_type,
            encoder=encoder
        )
        
        for k, v in kwargs.items():
            if k in ModelCreateArgs.__annotations__:
                setattr(model_create_args, k, v)
        
        return model_create_args

    @classmethod
    def by_id(
        cls, model_id: str | PydanticObjectId, fc: Any
    ) -> "FeatrixNeuralFunction":
        """
        Get a predictive model by its id.

        Returns:
            FeatrixNeuralFunction: The model if it exists, otherwise None.
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()
        result = fc.api.op("model_get", model_id=str(model_id))
        return ApiInfo.reclass(cls, result, fc=fc)

    @classmethod
    def from_job(
        cls, job: FeatrixJob, fc: Optional["Featrix"] = None
    ) -> "FeatrixNeuralFunction":  # noqa F821
        """
        Given a FeatrixJob, retrieve the predictive model that is referenced by the job.

        Arguments:
            job: FeatrixJob: The job that references the model -- must have model_id set

        Returns:
            FeatrixNeuralFunction: The model if it exists, otherwise None.
        """
        from .networkclient import Featrix

        if fc is None:
            fc = Featrix.get_instance()
        return cls.by_id(str(job.model_id), fc=fc)

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

    def ready(self):
        return self.training_state == TrainingState.COMPLETED

    def get_jobs(self, active: bool = True, training: bool = True) -> List[FeatrixJob]:
        """
        Return a list of jobs associated with this model.

        By default, only active (unfinished) training jobs are returned. Use the arguments to filter results.

        Args:
            active (bool): If `True`, return only active jobs.
            training (bool): If `True`, return only training jobs.

        Returns:
            List[FeatrixJob]: The list of jobs associated with this model.
        """

        jobs = []
        for job in FeatrixJob.by_neural_function(self):
            if active or training:
                if active and job.finished:
                    continue
                if training and job.job_type != JobType.JOB_TYPE_MODEL_CREATE:
                    continue
            jobs.append(job)
        return jobs

    def predict(self, query: Dict | List[Dict] | pd.DataFrame) -> FeatrixPrediction:
        """
        Predict a probability distribution on a given model in a embedding space.

        Query can be a list of dictionaries or a dictionary for a single query.

        Parameters
        ----------
        query : dict, [dict], or pd.DataFrame
            Either a single parameter or a list of parameters.
            { col1: <value> }, { col2: <value> }

        Returns
        -------
        A dictionary of values of the model's target_column and the probability of each of those values occurring.

        """
        if isinstance(query, dict):
            query = [query]
        if isinstance(query, pd.DataFrame):
            query = query.to_dict(orient="records")
        predict_args = ModelFastPredictionArgs(model_id=str(self.id), query=query)
        pred = self._fc.api.op("models_create_prediction", predict_args)
        fp = ApiInfo.reclass(FeatrixPrediction, pred, fc=self._fc)
        return fp.result
    
    @classmethod
    def new_neural_function(
        cls,
        fc: Any,
        target_field: str,
        embedding_space: "FeatrixEmbeddingSpace",
        project: "FeatrixProject",
        target_field_type: Optional[str] = 'auto', # set | scalar
        encoder: Optional[Dict] = None,
        wait_for_completion: bool = True,
        **kwargs,
    ) -> "FeatrixNeuralFunction":
        """
        Create a neural function. Generally you would use FeatrixEmbeddingSpace.create_neural_function(), which calls this.
        """
        from .featrix_job import FeatrixJob
        from .featrix_project import FeatrixProject  # noqa forward ref

        project = project.refresh()
        # before_nf_list = project.neural_functions()

        name = kwargs.pop("name", f"Predict_{'_'.join(target_field)}")
        
        if name is None:
            name = (
                f"{project.name}-{uuid.uuid4()}"
                if isinstance(project, FeatrixProject)
                else f"Project {uuid.uuid4()}"
            )

        nf_create_args = cls.create_args(
            str(project.id),
            embedding_space.id,
            target_field,
            target_field_type,
            encoder=encoder,
            **kwargs,
        )

        dispatch = fc.api.op("job_model_create", nf_create_args)
        job = FeatrixJob.from_job_dispatch(dispatch, fc)

        if wait_for_completion:
            job.wait_for_completion(f"Job {job.job_type} (job id={job.id}):")
            job = job.refresh()
            # print("------ job finished ----: ", job)
            if job.error:
                raise FeatrixException(
                    f"Failed to train neural function on target field '{target_field}': {job.error_msg}"
                )
            after_nf_list = project.neural_functions()
            for nf in after_nf_list:
                if str(nf.job_id) == str(job.id):
                    nf.project = project
                    return nf
            raise FeatrixException(f"We waited for completion for job id={job.id} but did not find the resulting function in the project id={project.id} afterwards. This is not expected.")
        else:
            # 4 Sept 2024 MH
            # Right now, we do not allocate the model_id on the server until the job is done.
            # Once we change that, we can improve this interface.
            print(f"Warning: If you do not set wait_for_completion to True, we don't have (currently) a good way to get back info later [you can query job_id = {job.id}]")
        return None
