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

import sys


def running_in_notebook():
    try:
        from IPython import get_ipython

        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except:  # noqa - anyway this fails, we aren't in a notebook
        return False
    return True


def clear_cell(wait=False):
    try:
        from IPython.display import clear_output

        if running_in_notebook():
            clear_output(wait=wait)
    except ImportError:  # noqa
        pass


def display_message(msg: str):
    clear_cell(wait=False)
    print(msg)


