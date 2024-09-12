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
    JOB_TYPE_MODEL_CREATE = "model-create"
    JOB_TYPE_MODEL_PREDICTION = "model-prediction"
    JOB_TYPE_MODEL_GUARDRAILS = "model-guardrails"
    # JOB_TYPE_AUTOJOIN_MAIN       = "autojoin-main"
    JOB_TYPE_ENCODE_RECORDS = "encode-records"
    JOB_TYPE_TEST_CASE = "test-case"
    JOB_TYPE_PROCESS_UPLOAD_DATA = "process-uploaded-data"
    JOB_TYPE_PROCESS_SMART_ENRICHMENT = "process-smart-enrichment"
    # JOB_TYPE_DETECT_ENCODERS     = "detect-encoders"
    JOB_TYPE_HAYSTACK_PROCESSING = "haystack-processing"
    JOB_TYPE_HAYSTACK_WAIT_PREP = "haystack-wait-prep"
    JOB_TYPE_HAYSTACK_RUN_MODEL = "haystack-run-model"

    def is_training_job_type(self):
        if self in [
            JobType.JOB_TYPE_ES_TRAIN_MORE,
            JobType.JOB_TYPE_ES_CREATE,
            JobType.JOB_TYPE_MODEL_CREATE,
        ]:
            return True
        return False


# class ChainedJobType(str, Enum):
#     CHAINED_JOB_TYPE_NNF = "chained-new-neural-function"
#     CHAINED_JOB_TYPE_NNF_WAIT_ES = "chained-new-neural-function-wait-es"
#     CHAINED_JOB_TYPE_NEW_EXPLORER = "chained-new-explorer"
#     # CHAINED_JOB_TYPE_HAYSTACK_PREDICTOR = "chained-haystack-predictor"
