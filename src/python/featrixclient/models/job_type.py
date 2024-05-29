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


class JobType(str, Enum):
    """
    JobType is set automatically in the JobArgs classes for submitting new jobs.  It will also
    appear in the JobMeta database object describing a job that was submitted.
    """

    JOB_TYPE_ES_TRAIN_MORE = "embedding-space-train-more"
    JOB_TYPE_ES_CREATE = "embedding-space-create"
    JOB_TYPE_ES_CREATE_PROJECTION = "embedding-space-projection-create"
    JOB_TYPE_ES_WAIT_TO_TRAIN = "embedding-space-waiting-for-training"
    JOB_TYPE_ES_CREATE_FROM_DS = "embedding-space-create-from-data-space"
    JOB_TYPE_ES_CREATE_DB = "embedding-space-create-database"
    JOB_TYPE_DB_CLUSTER = "vector-db-cluster"
    JOB_TYPE_DB_NN_QUERY = "vector-db-nn-query"
    JOB_TYPE_MODEL_CREATE = "model-create"
    JOB_TYPE_MODEL_PREDICTION = "model-prediction"
    JOB_TYPE_MODEL_GUARDRAILS = "model-guardrails"
    # JOB_TYPE_AUTOJOIN_MAIN       = "autojoin-main"
    JOB_TYPE_AUTOJOIN_DETECT = "autojoin-detect"
    JOB_TYPE_AUTOJOIN_PROJECTION = "autojoin-projection"
    JOB_TYPE_ENCODE_RECORDS = "encode-records"
    JOB_TYPE_TEST_CASE = "test-case"
    JOB_TYPE_PROCESS_UPLOAD_DATA = "process-uploaded-data"
    JOB_TYPE_PROCESS_SMART_ENRICHMENT = "process-smart-enrichment"
    # JOB_TYPE_DETECT_ENCODERS     = "detect-encoders"
    JOB_TYPE_ES_DISTANCE = "embedding-space-distance"
    JOB_TYPE_HAYSTACK_PROCESSING = "haystack-processing"


class ChainedJobType(str, Enum):
    CHAINED_JOB_TYPE_NNF = "chained-new-neural-function"
    CHAINED_JOB_TYPE_NNF_WAIT_ES = "chained-new-neural-function-wait-es"
    CHAINED_JOB_TYPE_NEW_EXPLORER = "chained-new-explorer"
