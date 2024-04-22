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

import signal
import sys


class CtrlcHandler:
    gotControlC = False
    gotActiveJob = None

    def __init__(self, job_id):
        signal.signal(signal.SIGINT, CtrlcHandler.ctrlc_handler)
        self.reset_control_c()
        CtrlcHandler.gotActiveJob = job_id

    def __del__(self):
        # print("CtrlcHandler __del__ called")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        CtrlcHandler.gotActiveJob = None

    @staticmethod
    def abort_loop():
        return CtrlcHandler.gotControlC

    @staticmethod
    def ctrlc_handler(signum, frame):
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n>>> OK... stopping watch. But the background job will continue!")
        print(f">>> job_id = '{CtrlcHandler.gotActiveJob}'")
        print(">>> Call .kill() to stop background training.")
        CtrlcHandler.gotControlC = True
        return

    @staticmethod
    def reset_control_c():
        CtrlcHandler.gotControlC = False
