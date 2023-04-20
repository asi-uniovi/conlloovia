"""This module contains the classes and methods to help heuristic algorithms."""

from typing import Dict

from .model import (
    App,
    Problem,
    Container,
    ContainerClass,
    InstanceClass,
    Vm,
)


class ProblemHelper:
    """This class provides helper methods to solve the problem."""

    def __init__(self, problem: Problem) -> None:
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem

    def create_vms_dict(self) -> Dict[str, Vm]:
        """Creates a dictionary of VMs, indexed by their name."""
        vms = {}
        for ic in self.problem.system.ics:
            for vm_num in range(ic.limit):
                new_vm = Vm(ic=ic, num=vm_num)
                vms[new_vm.name()] = new_vm

        return vms

    def create_containers_dict(self, vms: Dict[str, Vm]) -> Dict[str, Container]:
        """Creates a dictionary of containers, indexed by their name."""
        containers = {}
        for vm in vms.values():
            for cc in self.problem.system.ccs:
                new_container = Container(cc=cc, vm=vm)
                containers[new_container.name()] = new_container

        return containers

    def get_vms_ordered_by_cores_desc(self, vms: Dict[str, Vm]) -> list[Vm]:
        """Returns a list of VMs ordered by decreasing number of cores. If they
        have the same number of cores, it orders them by cost."""
        return sorted(
            vms.values(),
            key=lambda vm: (-vm.ic.cores, vm.ic.price),
        )

    def get_ccs_ordered_by_cores_asc(self) -> list[ContainerClass]:
        """Returns a list of container classes ordered by increasing number of
        cores and memory."""
        return sorted(
            self.problem.system.ccs,
            key=lambda cc: (cc.cores, cc.mem),
        )

    def compute_cheapest_ic(self) -> InstanceClass:
        """Returns the cheapest instance class in terms of cores per dollar.
        If there are several, select the one with the smallest number of
        cores."""
        return min(
            self.problem.system.ics,
            key=lambda ic: (
                ic.price.to("usd/h") / ic.cores,
                ic.cores,
            ),
        )

    def create_empty_vm_alloc(self, vms: dict[str, Vm]) -> Dict[Vm, bool]:
        """Creates a VM allocation where no VM is allocated."""
        vm_alloc: Dict[Vm, bool] = {}
        for vm in vms.values():
            vm_alloc[vm] = False

        return vm_alloc

    def create_empty_container_alloc(
        self, containers: dict[str, Container]
    ) -> Dict[Container, int]:
        """Creates a container allocation where no container is
        allocated."""
        container_alloc: Dict[Container, int] = {}
        for container in containers.values():
            container_alloc[container] = 0

        return container_alloc

    def get_ccs_for_app(self, app: App) -> tuple[ContainerClass]:
        """Returns a tuple of container classes for this app."""
        return tuple(cc for cc in self.problem.system.ccs if cc.app == app)

    def get_ccs_ordered_by_cores_and_mem(self, app: App) -> tuple[Container]:
        """Returns a tuple of container classes for this app ordered by increasing
        number of cores and memory."""
        return tuple(
            sorted(
                self.get_ccs_for_app(app),
                key=lambda cc: (cc.cores, cc.mem),
            )
        )
