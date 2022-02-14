"""Data classes for the container model of conlloovia"""

from typing import List, Dict, Tuple
from xmlrpc.client import Boolean

from pydantic import BaseModel


class HashableBaseModel(BaseModel):
    def __hash__(self):  # make hashable BaseModel subclass
        return hash((type(self),) + tuple(self.__dict__.values()))


class App(HashableBaseModel):
    name: str


class InstanceClass(HashableBaseModel):
    name: str
    price: float = 0
    cores: float = 0  # millicores
    mem: float = 0  # GiB


class ContainerClass(HashableBaseModel):
    name: str
    cores: float = 0
    mem: float = 0
    app: App


# TODO: Remove?
class Perf(BaseModel):
    ic: InstanceClass
    cc: ContainerClass
    value: float


class System(BaseModel):
    apps: List[App]
    ics: List[InstanceClass]
    ccs: List[ContainerClass]
    # perfs: List[Perf] # TODO
    perfs: Dict[Tuple[InstanceClass, ContainerClass], float]


class Workload(BaseModel):
    value: float
    app: App
    time_unit: str  # “y”, “h”, “m”, or “s”


class Problem(BaseModel):
    system: System
    workloads: Dict[App, Workload]


class Vm(HashableBaseModel):
    ic: InstanceClass
    num: int


class Container(HashableBaseModel):
    cc: ContainerClass
    vm: Vm
    num: int


class Allocation(BaseModel):
    vms: Dict[Vm, bool]
    containers: Dict[Container, bool]


class Solution(BaseModel):
    problem: Problem
    alloc: Allocation
    cost: float
