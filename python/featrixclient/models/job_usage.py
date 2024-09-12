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

import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import psutil
from pydantic import Field

from .fmodel import FModel

logger = logging.getLogger(__name__)


class PlatformInfo(FModel):
    kernel_version: Optional[str] = None
    system_name: Optional[str] = None
    node_name: Optional[str] = None
    machine: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.system_name is None:
            self.kernel_version = os.uname().release
            self.system_name = os.uname().sysname
            self.node_name = os.uname().nodename
            self.machine = os.uname().machine


class MemoryInfo(FModel):
    total_memory: float = 0.0
    available_memory: float = 0.0
    used_memory: float = 0.0
    memory_percentage: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        if self.total_memory == 0.0:
            self.total_memory = psutil.virtual_memory().total / (1024.0**3)
            self.available_memory = psutil.virtual_memory().available / (1024.0**3)
            self.used_memory = psutil.virtual_memory().used / (1024.0**3)
            self.memory_percentage = psutil.virtual_memory().percent


class CPUInfo(FModel):
    physical_cores: int = 0
    total_cores: int = 0
    cpu_usage_per_core: list = Field(default_factory=list)
    total_cpu_usage: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        if self.physical_cores == 0:
            self.physical_cores = psutil.cpu_count(logical=False)
            self.total_cores = psutil.cpu_count(logical=True)
            self.cpu_usage_per_core = psutil.cpu_percent(percpu=True, interval=1)
            self.total_cpu_usage = psutil.cpu_percent(interval=1)


class DiskInfo(FModel):
    total_space: float
    used_space: float
    free_space: float
    usage_percentage: float


class PartitionDetail(FModel):
    device: str
    mountpoint: str
    fstype: str
    opts: Optional[str] = None
    maxfile: Optional[int] = None
    maxpath: Optional[int] = None


class PartitionUsage(FModel):
    total: int
    used: int
    free: int
    percent: float


class PartitionInfo(FModel):
    partition_usage: PartitionUsage
    partition_detail: PartitionDetail
    disk_info: DiskInfo


class DiskOverview(FModel):
    partitions: List[PartitionInfo]

    @classmethod
    def create(cls):
        parts = []
        for partition in psutil.disk_partitions():
            usage = psutil.disk_usage(partition.mountpoint)
            p = PartitionInfo(
                partition_usage=PartitionUsage(
                    **psutil.disk_usage(partition.mountpoint)._asdict()
                ),
                partition_detail=PartitionDetail(**partition._asdict()),
                disk_info=DiskInfo(
                    total_space=usage.total / (1024.0**3),
                    used_space=usage.used / (1024.0**3),
                    free_space=usage.free / (1024.0**3),
                    usage_percentage=usage.percent,
                ),
            )
            parts.append(p)
        return cls(partitions=parts)

    @staticmethod
    def get_disk_usage(path: Path) -> Tuple[int, int]:
        cnt = size = 0
        for file in path.glob("*"):
            if file.is_dir():
                sub_cnt, sub_size = DiskOverview.get_disk_usage(file)
                size += sub_size
                cnt += sub_cnt
            elif file.is_file():
                size += file.stat().st_size
                cnt += 1
        return cnt, size


class LoadStats(FModel):
    load_average_1: float = -1.0
    load_average_5: float = -1.0
    load_average_15: float = -1.0

    def model_post_init(self, __context: Any) -> None:
        if self.load_average_1 == -1.0:
            self.load_average_1 = psutil.getloadavg()[0]
            self.load_average_5 = psutil.getloadavg()[1]
            self.load_average_15 = psutil.getloadavg()[2]


class ProcessStats(FModel):
    user: float = -1.0
    system: float = -1.0
    children_user: float = -1.0
    children_system: float = -1.0

    def model_post_init(self, __context: Any) -> None:
        if self.user == -1.0:
            self.user = psutil.Process().cpu_times().user
            self.system = psutil.Process().cpu_times().system
            self.children_user = psutil.Process().cpu_times().children_user
            self.children_system = psutil.Process().cpu_times().children_system


class JobUsageStats(FModel):
    hostname: str = socket.gethostname()
    taken: datetime = Field(default_factory=datetime.utcnow)
    cpu_usage: Optional[ProcessStats] = Field(default_factory=ProcessStats)
    platform: Optional[PlatformInfo] = Field(default_factory=PlatformInfo)
    memory: Optional[MemoryInfo] = Field(default_factory=MemoryInfo)
    cpu_hw: Optional[CPUInfo] = Field(default_factory=CPUInfo)
    disk: Optional[DiskOverview] = Field(default_factory=DiskOverview.create)
    load: Optional[LoadStats] = Field(default_factory=LoadStats)
    job_disk_usage: Optional[int] = None
    customer_disk_usage: Optional[int] = None
