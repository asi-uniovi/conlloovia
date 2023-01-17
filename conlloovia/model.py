"""Data classes for the container model of conlloovia"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum

import pint

ureg = pint.UnitRegistry()

# Define new units
ureg.define("usd = [currency]")
ureg.define("core = [computation]")
ureg.define("millicore = 0.001 core")
ureg.define("req = [requests]")


class Status(Enum):
    "Possible status of conlloovia solutions"
    UNSOLVED = 0
    OPTIMAL = 1
    INTEGER_FEASIBLE = 2
    INFEASIBLE = 3
    INTEGER_INFEASIBLE = 4
    OVERFULL = 5
    TRIVIAL = 6
    ABORTED = 7
    CBC_ERROR = 8
    UNKNOWN = 9


@dataclass(frozen=True)
class App:
    name: str


@dataclass(frozen=True)
class InstanceClass:
    name: str
    price: pint.Quantity  # [currency]/[time]
    cores: pint.Quantity  # [computation]
    mem: pint.Quantity  # dimensionless
    limit: int  # Max. number of VMs of this instance class

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        object.__setattr__(self, "price", self.price.to("usd/hour"))
        object.__setattr__(self, "cores", self.cores.to("cores"))
        object.__setattr__(self, "mem", self.mem.to("gibibytes"))


@dataclass(frozen=True)
class ContainerClass:
    name: str
    cores: pint.Quantity  # [computation]
    mem: pint.Quantity  # dimensionless
    app: App
    limit: int  # Max. number of containers of this container class

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        object.__setattr__(self, "cores", self.cores.to("cores"))
        object.__setattr__(self, "mem", self.mem.to("gibibytes"))


@dataclass(frozen=True)
class System:
    apps: Tuple[App, ...]
    ics: Tuple[InstanceClass, ...]
    ccs: Tuple[ContainerClass, ...]
    perfs: Dict[Tuple[InstanceClass, ContainerClass], pint.Quantity]

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        new_perfs = {}
        for key, value in self.perfs.items():
            new_perfs[key] = value.to("req/hour")

        object.__setattr__(self, "perfs", new_perfs)


@dataclass(frozen=True)
class Workload:
    num_reqs: float  # [req]
    time_slot_size: pint.Quantity  # [time]
    app: App

    def __post_init__(self):
        """Checks dimensions of the time_slot_size are valid."""
        self.time_slot_size.to(
            "hour"
        )  # If the dimensions are wrong, this raises an Exception


@dataclass(frozen=True)
class Problem:
    system: System
    workloads: Dict[App, Workload]
    sched_time_size: pint.Quantity  # Size of the scheduling window [time]

    def __post_init__(self):
        """Checks dimensions of the sched_time_size are valid. In addition, it
        must be the same as the workload time slot size for all workloads."""
        self.sched_time_size.to(
            "hour"
        )  # If the dimensions are wrong, this raises an Exception

        for wl in self.workloads.values():
            if wl.time_slot_size != self.sched_time_size:
                raise ValueError(
                    f"All workloads should have the time slot unit {self.sched_time_size}"
                )


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
class SolvingStats:
    frac_gap: Optional[float]
    max_seconds: Optional[float]
    lower_bound: Optional[float]
    creation_time: float
    solving_time: float
    status: Status


@dataclass(frozen=True)
class Solution:
    problem: Problem
    alloc: Allocation
    cost: pint.Quantity  # [currency]
    solving_stats: SolvingStats

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        object.__setattr__(self, "cost", self.cost.to("usd"))
