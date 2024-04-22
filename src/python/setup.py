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
from typing import Optional

from setuptools import find_packages
from setuptools import setup

name = "featrixclient"
description = "Featrix AI API"
entry_points = None
home = "https://featrix.ai"
excludes = ["dist", "dist.*", "__pycache__", "*.ipynb", "*.ipynb_checkpoints", "*.log"]
current = Path(__file__).parent


def read_release(file: Optional[Path] = None):
    if file is None:
        file = Path(__file__).parent / "VERSION"
    try:
        data = file.read_text()
        _major, _minor, _iteration = data.split(".")
    except FileNotFoundError:
        now = datetime.now()
        _major, _minor, _iteration = int(now.year), int(f"{now.month}{now.day:02}"), 1
        file.write_text(f"{_major}.{_minor}.{_iteration}\n")
    return int(_major), int(_minor), int(_iteration)


major, minor, iteration = read_release()

setup(
    name=name,
    version=f"{major}.{minor}.{iteration}",
    python_requires=">=3.10",
    description=description,
    long_description=(current / "README.md").read_text(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Data Scientists",
        "License :: Other/Proprietary",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Installation/Setup",
        "Topic :: System :: System Administration",
        "Topic :: Utilities",
    ],
    keywords=[
        "ML",
        "AI",
        "embedding",
        "embeddings",
        "tabular",
        "embeddings",
        "vectorize",
        "automl",
    ],
    # ?
    download_url="https://github.com/Featrix/opensource.git",
    url="https://featrix.com",
    author="Featrix, Inc.",
    author_email="hello@featrix.ai",
    license=(current / "LICENSE").read_text(),
    install_requires=(current / "requirements.txt").read_text().split("\n"),
    packages=find_packages(exclude=excludes, where="."),
    package_dir={"featrixclient": "featrixclient"},
    # include_package_data=True,
    # exclude_package_data=excludes,
    entry_points=entry_points,
)
