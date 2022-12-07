"""Data classes for the container model of conlloovia"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum


class Status(Enum):
    "Possible status of conlloovia solutions"
    UNSOLVED = 0
    OPTIMAL = 1
    INFEASIBLE = 2
    INTEGER_INFEASIBLE = 3
    OVERFULL = 4
    TRIVIAL = 5
    ABORTED = 6
    CBC_ERROR = 7
    UNKNOWN = 8


@dataclass(frozen=True)
class App:
    name: str


@dataclass(frozen=True)
class InstanceClass:
    name: str
    price: float
    cores: float  # millicores
    mem: float  # GiB
    limit: int


@dataclass(frozen=True)
class ContainerClass:
    name: str
    cores: float
    mem: float
    app: App
    limit: int


@dataclass(frozen=True)
class System:
    apps: List[App]
    ics: List[InstanceClass]
    ccs: List[ContainerClass]
    perfs: Dict[Tuple[InstanceClass, ContainerClass], float]


@dataclass(frozen=True)
class Workload:
    value: float
    app: App
    time_unit: str  # “y”, “h”, “m”, or “s”


@dataclass(frozen=True)
class Problem:
    system: System
    workloads: Dict[App, Workload]


@dataclass(frozen=True)
class Vm:
    ic: InstanceClass
    num: int


@dataclass(frozen=True)
class Container:
    cc: ContainerClass
    vm: Vm
    num: int


@dataclass(frozen=True)
class Allocation:
    vms: Dict[Vm, bool]
    containers: Dict[Container, bool]


@dataclass(frozen=True)
class Solution:
    problem: Problem
    alloc: Allocation
    cost: float
    status: Status
