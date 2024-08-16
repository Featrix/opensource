#!/usr/bin/env python3
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
from datetime import datetime
from pathlib import Path


def write_py_version(file, major, minor, iteration):
    file.write(f'version = "{major}.{minor}.{iteration}"\n')
    file.write(f'publish_time = "{datetime.utcnow().isoformat()}"\n')
    file.write('__author__ = "Featrix, Inc."\n')


def increment_version():
    home = Path(__file__).parent
    version_path = home / "VERSION"
    py_version_path = home / "featrixclient/version.py"

    new_major = datetime.now().year
    new_minor = datetime.now().month * 100 + datetime.now().day
    with version_path.open("r") as f:
        version = f.read().strip().split(".")
        major, minor, iteration = int(version[0]), int(version[1]), int(version[2])
        if major != new_major or minor != new_minor:
            major, minor, iteration = new_major, new_minor, 1
        else:
            iteration += 1
    version_path.write_text(f"{major}.{minor}.{iteration}")

    py_version = py_version_path.read_text().split("\n")
    with py_version_path.open("w") as _f:
        for line in py_version:
            if line.startswith("#"):
                _f.write(line + "\n")
            else:
                write_py_version(_f, major, minor, iteration)
                break
        else:
            write_py_version(_f, major, minor, iteration)


if __name__ == "__main__":
    increment_version()
