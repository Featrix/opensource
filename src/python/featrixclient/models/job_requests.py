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

from enum import Enum
from typing import Annotated, Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BeforeValidator
from pydantic import Field

from .fmodel import FModel
from .job_type import ChainedJobType
from .job_type import JobType
from .pydantic_objectid import PydanticObjectId

#
# Some types that help convert input into proper forms whether they are coming from an API call or
# a manual argument if running tasks by hand

#
# A list of one or more Pydantic ids, possibly as strings, separated by commas
#
PydanticObjectIdList = Annotated[
    List[PydanticObjectId],
    BeforeValidator(
        lambda v: [PydanticObjectId(_) for _ in v]
        if isinstance(v, list)
        else [PydanticObjectId(_) for _ in v.replace(" ", "").split(",")]
    ),
]
#
# A list of string tokens separated by commas if multiple
#
StrList = Annotated[
    List[str],
    BeforeValidator(
        lambda v: [_.strip() for _ in v.split(",")]
        if isinstance(v, str)
        else [str(_).strip() for _ in v]
    ),
]

#
# A dictionary of key/values, possibly specified in a comma separated list like "key=value,key2=value2"
#
InputDict = Annotated[
    Dict,
    BeforeValidator(
        lambda v: v
        if isinstance(v, dict)
        else {a.strip(): b.strip() for a, b in [_.split("=") for _ in v.split(",")]}
    ),
]
# Is a list of dicts, but caller might just supply one dict stand alone
ListOfDicts = Annotated[
    List[Dict], BeforeValidator(lambda v: [v] if isinstance(v, dict) else v)
]


class JobArgs(FModel):
    job_type: JobType
    skip_auto_load: Optional[bool] = False


class ChainedJobArgs(FModel):
    job_type: ChainedJobType


# Where should this go?
class Encoders(Enum):
    ENCODER_FREE_STRING = "free_string"
    ENCODER_SET = "set"
    ENCODER_SCALAR = "scalar"
    ENCODER_LISTS_SET = "lists_of_a_set"

    @classmethod
    def valid(cls, key: str) -> bool:
        if key in [
            cls.ENCODER_FREE_STRING,
            cls.ENCODER_SET,
            cls.ENCODER_SCALAR,
            cls.ENCODER_LISTS_SET,
        ]:
            return True
        return False


class ESWaitToFinish(JobArgs):
    """
    Use this when you want to queue another job in a chain after an embedding space has finished training.
    """
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ES_WAIT_TO_TRAIN,
        frozen=True,
        description="A Embedding Space Creation Job. The JobType should not be changed",
    )

    #  The ES we are waiting for -- this is required!
    embedding_space_id: PydanticObjectId
    other_job_id: PydanticObjectId

    # do not autoload stuff into the job work area.
    skip_auto_load: bool = True

    # the model who is waiting.
    model_id: Optional[PydanticObjectId] = None

    # Schedule model builds upon completion?
    build_explorer_models: bool = Field(default=False)


class ESCreateArgs(JobArgs):
    """
    This is used for creating a new embedding space.  The embedding space is created inside a project,
    and will use the associated files in that project for the training.  Most of the arguments, aside
    from the project identifier, have defaults but can be overridden.
    """

    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ES_CREATE,
        frozen=True,
        description="A Embedding Space Creation Job. The JobType should not be changed",
    )
    # Can provide either a list of upload ids OR a project id
    upload_ids: Optional[List[PydanticObjectId]] = None
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description=" The ID of the project in which to create an embedding space",
        title="Project ID",
    )
    # If we already have an skeleton embedding space in the db, pass it down.
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="If the embedding space was pre-allocated, the id of the DB object",
        title="Embedding Space ID",
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the Embedding Space, must be unique in your organization",
        title="Name of the Embedding Space",
    )
    #   Overrides the initial training epochs. FIXME: what is the default here?
    epochs: int = Field(
        default=5,
        description="The number times that the learning algorithm will work through the entire training dataset.",
        title="Training Epochs",
    )
    batch_size: int = Field(
        default=32,
        description="The number of samples to process when updating the model (breaking an epoch into smaller batches)",
    )
    # Run detection code and print out diagnostics and exit. No training of a embedding_space space.
    detect_only: bool = False
    detect_only_filename: Optional[str] = None
    # Overwrites an existing vector space if it exists.
    use_gpu: bool = Field(
        default=False,
        description="Whether to use GPU acceleration",
        title="Use GPU acceleration",
    )
    # Ignored the specified columns from the data set
    ignore_cols: Optional[StrList] = Field(
        default_factory=list,
        description="A list of column names to ignore in the dataset(s) used for training this embedding space",
        title="Ignore Column Names",
    )
    # Drop all columns except for the ones specified
    focus_cols: Optional[StrList] = Field(
        default_factory=list,
        description="A list of columns on which to focus the training",
        title="Focus Column Names",
    )
    debug_detector: Optional[str] = Field(
        default=None,
        description="Pass a detector name to debug it.",
    )
    # Limit the number of rows to ingest to the passed number.
    rows: Optional[int] = Field(
        default=None,
        description="Limit the number of rows to ingest into the training",
        title="Row Limit",
    )
    # Specify an encoder for a specific column -- this can be a string - comma separated col_name=encoder or
    # a dict (key is col_name, val is encoder)
    encoder: Optional[InputDict] = Field(
        default_factory=dict,
        description="A list of specific encoders to use for specific columns.  This can be a string "
        "that is a comma-separated list of col_name=encoder, or a dictionary where the "
        "keys are column names and the values are encoder names",
        title="Encoder Overrides",
    )
    #
    force: bool = Field(
        default=False,
        description="Whether to force the creation of an embedding space",
        title="Force creation",
    )
    # Set the learning rate. Default is 0.001.
    learning_rate: float = Field(
        description="The learning rate to use as a base starting point, which controls "
        "how quickly the model is adapted to the problem",
        title="Learning Rate",
        default=0.001,
    )
    # Drop duplicates true/false.
    drop_duplicates: bool = Field(
        default=True,
        description="Whether to drop duplicat columns in the training set",
    )
    # Sample percentage. Default is 1.0.
    sample_percentage: float = Field(
        default=1.0,
        description="The percentage of the input rows to use for training, specified as a percentage (1.0 = 100%)",
        title="Sampling Percentage",
    )
    training_debug: bool = False
    follow_on_model_info: Optional[Dict] = None
    training_budget_credits: Optional[float] = Field(
        default=None,
        description="Limit the training to consume only this many credits at most.",
        title="Budget for training",
    )
    # training_credit_budget: Optional[int] = None
    # preallocated_embedding_space_record_id: Optional[PydanticObjectId]


class ESCreateProjectionArgs(JobArgs):
    """
    Create a projection json from the embedding space given
    """

    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ES_CREATE_PROJECTION,
        frozen=True,
        description="A Embedding Space Projection Creation Job. The JobType should not be changed",
    )
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description=" The ID of the project in which the targeted embedding space exists",
        title="Project ID",
    )
    # If we already have an skeleton embedding space in the db, pass it down.
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The Embedding Space id of the Embedding Space from which to create the projection",
        title="Embedding Space ID",
    )


class HaystackProcessingArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_HAYSTACK_PROCESSING,
        frozen=True,
        description="Run Haystack processing on a project.",
    )
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project",
        title="Project ID",
    )
    config: Optional[dict] = None


class TrainMoreArgs(JobArgs):
    """
    This job is used to continue the training of an embedding space.
    """

    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ES_TRAIN_MORE,
        frozen=True,
        description="A Train Embedding Space More Job. The JobType should not be changed",
    )
    # We can get project id from embedding space id so the first is optional
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project to which this embedding space belongs, "
        "can be derived from embedding space",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        description="The ID of the embedding space we are continuing to train",
        title="Embedding Space ID",
    )
    epochs: int = Field(
        default=25,
        description="The number times that the learning algorithm will work through the entire training dataset.",
        title="Training Epochs",
    )
    batch_size: int = Field(
        default=32,
        description="The number of samples to process when updating the model (breaking an epoch into smaller batches)",
    )
    learning_rate: float = Field(
        description="The learning rate to use as a base starting point, which controls "
        "how quickly the model is adapted to the problem",
        title="Learning Rate",
        default=0.001,
    )
    training_debug: bool = Field(
        description="Whether to include extra debugging information in the training output",
        title="Training debugging",
        default=False,
    )
    training_budget_credits: Optional[float] = Field(
        default=None,
        description="Limit the training to consume only this many credits at most.",
        title="Budget for training",
    )


class EmbeddingsDistanceArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ES_DISTANCE,
        frozen=True,
        description="Embedding space distance. The JobType should not be changed",
    )
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project to which the embedding space belongs",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space in which compute distances",
        title="Embedding Space ID",
    )
    col_pairs: Optional[List[List]] = Field(
        default=None,
        description="The list of pairs we want to compute distances on.",
        title="Column pairs",
    )


class ModelCreateArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_MODEL_CREATE,
        frozen=True,
        description="A Create Model Job. The JobType should not be changed",
    )
    # We can get project id from embedding space id so the first is optional
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project to which this model belongs",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space in which to create the model",
        title="Embedding Space ID",
    )
    # If the caller pre-allocates the model
    model_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="If the model was pre-allocated, the id of the DB object",
        title="Model ID",
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the Model, must be unique within its Embedding Space",
        title="Name of the Model",
    )
    upload_ids: PydanticObjectIdList = Field(
        default_factory=list,
        description="A list of datasets in your library to use for creating the model",
        title="Upload IDs",
    )
    target_columns: StrList = Field(default_factory=list)
    epochs: int = Field(
        default=25,
        description="The number of epochs or cycles to initially train this model",
        title="Training Epochs",
    )
    learning_rate: float = Field(
        description="The learning rate to use as a base starting point, which controls "
        "how quickly the model is adapted to the problem",
        title="Learning Rate",
        default=0.0001,
    )
    model_size: str = Field(
        default="small",
        description="The size of the model, can be 'small' or 'large'",
        title="Model Size",
    )
    training_budget_credits: Optional[float] = Field(
        default=None,
        description="Limit the training to consume only this many credits at most.",
        title="Budget for training",
    )


class NewNeuralFunctionArgs(ChainedJobArgs):
    job_type: ChainedJobType = Field(
        default=ChainedJobType.CHAINED_JOB_TYPE_NNF,
        frozen=True,
        description="Create a New Neural Function by training an embedding space, and then training a "
        "model within that embedding space",
    )
    project_id: PydanticObjectId
    training_credits_budgeted: float
    embedding_space_create: ESCreateArgs
    model_create: ModelCreateArgs


class NewExplorerArgs(ChainedJobArgs):
    job_type: ChainedJobType = Field(
        default=ChainedJobType.CHAINED_JOB_TYPE_NEW_EXPLORER,
        frozen=True,
        description="Create an Explorer space by training an embedding space, and then training a "
        "model within that embedding space for each column",
    )
    project_id: PydanticObjectId
    training_credits_budgeted: float
    embedding_space_create: ESCreateArgs


class ModelPredictionArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_MODEL_PREDICTION,
        frozen=True,
        description="A Prediction on Model Job. The JobType should not be changed",
    )
    # We can get the project and embedding_space from the model
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project to which this model belongs, can be derived from model id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space that contains the model, can be derived from the model id",
        title="Embedding Space ID",
    )
    model_id: Optional[PydanticObjectId] = Field(
        description="The ID of the model to use for the prediction",
        title="Embedding Space ID",
    )
    query: ListOfDicts = Field(
        description="A list of queries to perform, each query is a dictionary with the column name as the key.",
        title="Queries",
    )


class ModelFastPredictionArgs(FModel):
    """
    This supports doing fast predictions -- it bypasses the requiring us to read in the embedding space,
    and project... This probably replaces the above arg but for now we made a separate one
    """

    job_type: JobType = Field(
        default=JobType.JOB_TYPE_MODEL_PREDICTION,
        frozen=True,
        description="A Prediction on Model Job. The JobType should not be changed",
    )
    # We can get the project and embedding_space from the model
    model_id: Optional[PydanticObjectId] = Field(
        description="The ID of the model to use for the prediction",
        title="Embedding Space ID",
    )
    query: ListOfDicts = Field(
        description="A list of queries to perform, each query is a dictionary with the column name as the key.",
        title="Queries",
    )


class GuardRailsArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_MODEL_GUARDRAILS,
        frozen=True,
        description="A Check Guardrails on prediction Job. The JobType should not be changed",
    )
    # We can get the project and embedding_space from the model
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project to which this model belongs, can be derived from model id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space that contains the model, can be derived",
        title="Embedding Space ID",
    )
    model_id: Optional[PydanticObjectId] = Field(
        description="The ID of the model to use for the guardrail check",
        title="Embedding Space ID",
    )
    issues_only: bool = Field(
        default=False,
        description="Whether to only return real issues, or to include all the checks performed without error",
        title="Issues Only",
    )
    query: ListOfDicts = Field(
        description="A list of queries to validate, each query is a dictionary with the column name as the key.",
        title="Queries",
    )


class EncodeRecordsArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ENCODE_RECORDS,
        frozen=True,
        description="An Encode Records Into Embedding Space Job. The JobType should not be changed",
    )
    # We can get the project from the embedding_space space
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project to which this model belongs, can be derived from embedding space id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space into which to encode the records",
        title="Embedding Space ID",
    )
    # We MIGHT want to encode an upload id... but we might not too.
    # Loading the upload from S3 is too heavy for the fast path
    # so I am commenting this out for now. Not sure the blast radius of this
    # breakage.
    #
    # upload_id: Optional[PydanticObjectId] = Field(
    #     description="The ID of the data set to encode into the embedding space",
    # )

    records: Any        # data to encode.

class AutoJoinDetectArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_AUTOJOIN_DETECT,
        frozen=True,
        description="An AutoDetect Join Tables Job. The JobType should not be changed",
    )
    # We can get the project from the embedding_space space
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project in which this autojoin happens, can be derived from embedding space id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space on which to do the autojoin operation",
        title="Embedding Space ID",
    )


class AutoJoinProjectionArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_AUTOJOIN_PROJECTION,
        frozen=True,
        description="An AutoDetect Projection Job. The JobType should not be changed",
    )
    # We can get the project from the embedding_space space
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project in which this autojoin happens, can be derived from embedding space id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space on which to do the autojoin projection",
        title="Embedding Space ID",
    )


class CreateDBArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ES_CREATE_DB,
        frozen=True,
        description="A Create Database from Embedding Space Job. The JobType should not be changed",
    )
    # We can get project id from embedding space id so the first is optional
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the project in which this DB is created, can be derived from embedding space id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space to use for creating the database",
        title="Embedding Space ID",
    )
    # FIXME: missing


class CreateFromDSArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_ES_CREATE_FROM_DS,
        frozen=True,
        description="FIXME:A Embedding Space Job. The JobType should not be changed",
    )
    # We can get project id from embedding space id so the first is optional
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description=" The ID of the project in which FIXME:, can be derived from embedding space id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space to use for creating the database",
        title="Embedding Space ID",
    )


class DBClusterArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_DB_CLUSTER,
        frozen=True,
        description="An Cluster Database Job. The JobType should not be changed",
    )
    # We can get project id from embedding space id so the first is optional
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description=" The ID of the project in which FIXME:, can be derived from embedding space id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space to use for FIXME: clustering database",
        title="Embedding Space ID",
    )


class NNQueryArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_DB_NN_QUERY,
        frozen=True,
        description="An Nearest Neighbor Query in Database Job. The JobType should not be changed",
    )
    # We can get project id from embedding space id so the first is optional
    project_id: Optional[PydanticObjectId] = Field(
        default=None,
        description=" The ID of the project in which we are operating, can be derived from embedding space id",
        title="Project ID",
    )
    embedding_space_id: Optional[PydanticObjectId] = Field(
        default=None,
        description="The ID of the embedding space to use for the query",
        title="Embedding Space ID",
    )


class TestCaseArgs(JobArgs):
    # Most of these are just examples, but you can ask the task to fail, or to take some number of seconds.
    # please_fail can be None (success), "exception" (fails with exception), "fail: {msg}" with the msg you want to
    # fail with.
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_TEST_CASE,
        frozen=True,
        description="This is a testing job, it does nothing. The JobType should not be changed",
    )
    project_id: PydanticObjectId
    please_fail: Optional[str] = None
    run_duration: int = 10
    name: str
    column_count: int
    column_name: str
    ignore_columns: List[str]
    ignore_errors: bool = False


class ProcessUploadArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_PROCESS_UPLOAD_DATA,
        frozen=True,
        description="Pre-process uploaded data. The JobType should not be changed",
    )

    upload_id: Optional[PydanticObjectId] = None

    # is this it?


class ProcessSmartEnrichmentArgs(JobArgs):
    job_type: JobType = Field(
        default=JobType.JOB_TYPE_PROCESS_SMART_ENRICHMENT,
        frozen=True,
        description="Pre-process uploaded data. The JobType should not be changed",
    )

    upload_id: Optional[PydanticObjectId] = None

    # is this it?
