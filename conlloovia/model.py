"""Data classes for the container model of conlloovia"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum

from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    CurrencyPerTime,
    Time,
    RequestsPerTime,
    Storage,
    Requests,
)


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
    """Represents an application."""

    name: str


@dataclass(frozen=True)
class InstanceClass:
    """Represents an instance class, i.e., a type of VM in a region, with its
    price and limits."""

    name: str
    price: CurrencyPerTime
    cores: ComputationalUnits
    mem: Storage
    limit: int  # Max. number of VMs of this instance class

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        object.__setattr__(self, "price", self.price.to("usd/hour"))
        object.__setattr__(self, "cores", self.cores.to("cores"))
        object.__setattr__(self, "mem", self.mem.to("gibibytes"))


@dataclass(frozen=True)
class ContainerClass:
    """Represents a container class, i.e., a type of container running an app
    with some resources."""

    name: str
    cores: ComputationalUnits
    mem: Storage
    app: App
    limit: int  # Max. number of containers of this container class

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        object.__setattr__(self, "cores", self.cores.to("cores"))
        object.__setattr__(self, "mem", self.mem.to("gibibytes"))


@dataclass(frozen=True)
class System:
    """Represents a system, i.e., a set of apps, instance classes, container
    classes, and performance values."""

    apps: Tuple[App, ...]
    ics: Tuple[InstanceClass, ...]
    ccs: Tuple[ContainerClass, ...]
    perfs: Dict[Tuple[InstanceClass, ContainerClass], RequestsPerTime]

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        new_perfs = {}
        for key, value in self.perfs.items():
            new_perfs[key] = value.to("req/hour")

        object.__setattr__(self, "perfs", new_perfs)


@dataclass(frozen=True)
class Workload:
    """Represents the workload for an app in a time slot."""

    num_reqs: Requests
    time_slot_size: Time
    app: App

    def __post_init__(self):
        """Checks that the units are valid."""
        # If the dimensions are wrong, this raises an Exception
        self.num_reqs.to("req")
        self.time_slot_size.to("hour")


@dataclass(frozen=True)
class Problem:
    """Represents a problem, i.e., a system, a set of workloads, and a
    scheduling time size."""

    system: System
    workloads: Dict[App, Workload]
    sched_time_size: Time  # Size of the scheduling window [time]

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
    """Represents a VM, which has an Instace class and a number to identify this
    VM in the list of instance classes."""

    ic: InstanceClass
    num: int

    def name(self) -> str:
        return f"{self.ic.name}-{self.num}"


@dataclass(frozen=True)
class Container:
    """Represents a set of container replicas running on a VM. It has a
    Container class and the VM where they are running."""

    cc: ContainerClass
    vm: Vm

    def name(self) -> str:
        return f"{self.vm.name()}-{self.cc.name}"


@dataclass(frozen=True)
class Allocation:
    """Represents an allocation. It has a dictionary, vm, where the keys are the
    VMs and the values are true or false, indicating if the VM is allocated or
    not. It also has a dictionary, containers, where the keys are the containers
    and the values are ints, indicating how many replicas of the container are
    allocated on the VM."""

    vms: Dict[Vm, bool]
    containers: Dict[Container, int]


@dataclass(frozen=True)
class SolvingStats:
    """Represents the solving statistics of a solution."""

    frac_gap: Optional[float]
    max_seconds: Optional[float]
    lower_bound: Optional[float]
    creation_time: float
    solving_time: float
    status: Status


@dataclass(frozen=True)
class Solution:
    """Represents a solution, i.e., an allocation and its cost. It also has
    the problem that was solve and the solving statistics."""

    problem: Problem
    alloc: Allocation
    cost: Currency
    solving_stats: SolvingStats

    def __post_init__(self):
        """Checks dimensions are valid and store them in the standard units."""
        object.__setattr__(self, "cost", self.cost.to("usd"))
