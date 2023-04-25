"""This module defines FirstFitAllocator, which is a simple allocator that
works like this:

- Create a list of VMs ordered by decreasing number of cores. This is the list
  of machines that we will try to fit the containers into.
- For each VM, create a list of containers ordered by increasing size of cores
  and memory.
- Order the apps by increasing size of their smallest container.
- For each app, allocate the containers in the order of the list of VMs and
  containers inside each VM. If we run out of VMs, the allocation is not
  feasible.
- After a feasible allocation is found, we try to improve it by moving the
  containers in the last VM to the cheapest VM that can fit them and supports
  the workload.
"""

import logging
import time
from typing import Optional

from cloudmodel.unified.units import Currency, ComputationalUnits, Storage, Requests

from .model import (
    Problem,
    ContainerClass,
    App,
    Allocation,
    Vm,
    Container,
    Solution,
    Status,
    SolvingStats,
)
from .problem_helper import ProblemHelper

logging.basicConfig(level=logging.DEBUG)


class ReplicaInfo:
    """Stores information about the number of replicas of a container class in
    a vm."""

    def __init__(self, container_class: ContainerClass) -> None:
        """Constructor.

        Args:
            container_class: container class"""
        self.container_class = container_class
        self.num_replicas = 0

    def __str__(self) -> str:
        """String representation."""
        return f"ReplicaInfo({self.container_class.name}, {self.num_replicas})"


class FirstFitAllocatorState:
    """State of the FirstFitAllocator while building a solution."""

    def __init__(self) -> None:
        # Number of allocated requests of the current app
        self.reqs_allocated = Requests("0 req")

        # Number of requests to allocate for the current app
        self.reqs_to_allocate = Requests("0 req")


class FirstFitAllocator:
    """First fit allocator."""

    def __init__(self, problem: Problem) -> None:
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem

        start_creation = time.perf_counter()

        self.helper = ProblemHelper(problem)

        self.vms: dict[str, Vm] = self.helper.create_vms_dict()
        self.containers: dict[str, Container] = self.helper.create_containers_dict(
            self.vms
        )

        self.ordered_vms: list[Vm] = self.helper.get_vms_ordered_by_cores_desc(self.vms)

        # Create a dictionary where the index is a VM and the value is a list of
        # ReplicaInfo objects, ordered by increasing number of cores and memory.
        ordered_ccs = self.helper.get_ccs_ordered_by_cores_asc()
        self.replica_info: dict[Vm, list[ReplicaInfo]] = {}
        for vm in self.ordered_vms:
            replica_info_list = []
            for cc in ordered_ccs:
                replica_info_list.append(ReplicaInfo(cc))
            self.replica_info[vm] = replica_info_list

        end_creation = time.perf_counter()
        self.creation_time = end_creation - start_creation

        self.start_solving = 0.0
        self.no_alloc_found = False

    def print_replica_info(self, compact_mode: bool) -> None:
        """Print the current situation about the replicas. If compact_mode is
        True, only the VMs that are used and the containers that have replicas
        are printed."""
        for vm in self.ordered_vms:
            if compact_mode and not self.is_vm_used(vm):
                continue

            print(f"VM {vm.name()} ({vm.ic.cores} cores, {vm.ic.mem})")
            for rep_info in self.replica_info[vm]:
                if compact_mode and rep_info.num_replicas == 0:
                    continue

                print(
                    f"  {rep_info.container_class.name} -> "
                    f"{rep_info.num_replicas} replicas"
                )

    def solve(self) -> Solution:
        """Solve the problem.

        Returns:
            Solution to the problem"""
        self.__find_feasible_alloc()

        if not self.no_alloc_found:
            self.__optimize_alloc()

        return self.__create_sol()

    def __find_feasible_alloc(self):
        """Find a feasible allocation using the larger VMs first."""

        self.start_solving = time.perf_counter()
        self.no_alloc_found = False  # We don't know until we try

        # We want to allocate the apps with the smallest container first
        for app in self.helper.get_apps_ordered_by_container_size_asc():
            self.__allocate_app(app)

            if self.no_alloc_found:
                break

    def is_vm_used(self, vm: Vm) -> bool:
        """Return True if the VM is used, i.e., if any of its replicas has a
        value greater than 1."""
        for replica_info in self.replica_info[vm]:
            if replica_info.num_replicas > 0:
                return True
        return False

    def __optimize_alloc(self) -> None:
        """Optimize the allocation by trying to fit the containers from the last
        machine into smaller machines."""
        last_used_vm = self.__find_last_used_vm()

        if not last_used_vm:
            # No VM is used
            return

        # Compute the number of cores and memory used in the last VM
        cores_used = ComputationalUnits("0 cores")
        mem_used = Storage("0 GB")
        for ri_src in self.replica_info[last_used_vm]:
            cores_used += ri_src.num_replicas * ri_src.container_class.cores
            mem_used += ri_src.num_replicas * ri_src.container_class.mem

        # Check if we can fit the containers from the last VM into a cheaper VM
        candidate_vms = self.ordered_vms[self.ordered_vms.index(last_used_vm) + 1 :]

        # Remove VMs which have a higher price than the last used VM
        candidate_vms = [
            vm for vm in candidate_vms if vm.ic.price < last_used_vm.ic.price
        ]

        # Order the candidate VMs by increasing price
        candidate_vms.sort(key=lambda vm: vm.ic.price)

        # Look for one VM where we can fit the containers from the last VM
        for vm in candidate_vms:
            if not (vm.ic.cores >= cores_used and vm.ic.mem >= mem_used):
                continue  # It doesn't fit

            self.__move_containers(last_used_vm, vm)

            # Check that the allocation is still feasible
            if self.__is_feasible_alloc():
                return

            # It's not feasible, so undo the changes
            logging.debug(
                "The allocation is not feasible after moving the containers "
                "from the last VM. Undoing the changes..."
            )
            self.__move_containers(vm, last_used_vm)

        logging.debug(
            "No cheaper VM that can fit the containers from the last VM"
            " while maintaining a feasible allocation."
        )

    def __is_feasible_alloc(self) -> bool:
        """Return True if the current allocation is feasible."""
        for app in self.problem.system.apps:
            if not self.__is_feasible_alloc_app(app):
                return False
        return True

    def __is_feasible_alloc_app(self, app: App) -> bool:
        """Return True if the current allocation of the app is feasible."""
        return self.__perf_alloc_app(app) >= self.problem.workloads[app].num_reqs

    def __perf_alloc_app(self, app) -> Requests:
        """Return the performance of the allocation for the app."""
        perfs = self.problem.system.perfs
        perf_app = Requests("0 req")
        for vm, l_ri in self.replica_info.items():
            for rep_info in l_ri:
                if rep_info.container_class.app == app:
                    perf_app += (
                        rep_info.num_replicas
                        * perfs[vm.ic, rep_info.container_class]
                        * self.problem.sched_time_size
                    )
        return perf_app

    def __move_containers(self, src: Vm, dst: Vm) -> None:
        """Move the containers from the source VM to the destination VM."""
        for ri_src in self.replica_info[src]:
            self.__move_replica(src, dst, ri_src)

    def __move_replica(self, last_used_vm: Vm, vm: Vm, ri_src: ReplicaInfo) -> None:
        """Move a replica from the last used VM to the new VM."""
        if ri_src.num_replicas == 0:
            return  # Nothing to move

        for ri_dst in self.replica_info[vm]:
            if ri_dst.container_class != ri_src.container_class:
                continue  # Different container class

            logging.debug(
                "Moving %d replicas of %s from %s to %s",
                ri_src.num_replicas,
                ri_src.container_class.name,
                last_used_vm.name(),
                vm.name(),
            )
            ri_dst.num_replicas = ri_src.num_replicas
            ri_src.num_replicas = 0
            break

    def __find_last_used_vm(self) -> Optional[Vm]:
        last_used_vm = None
        for vm in self.ordered_vms:
            if self.is_vm_used(vm):
                last_used_vm = vm
            else:
                break
        return last_used_vm

    def __allocate_app(self, app: App) -> None:
        """Allocate enough containers of the app to satisfy the workload.

        Args:
            app: app to allocate
            state: state of the allocator"""
        state = FirstFitAllocatorState()
        state.reqs_allocated = Requests("0 reqs")
        state.reqs_to_allocate = self.problem.workloads[app].num_reqs

        for vm in self.ordered_vms:
            logging.debug("Allocating app %s in vm %s", app.name, vm.name())
            self.allocate_replicas_in_vm(app, vm, state)
            if state.reqs_allocated >= state.reqs_to_allocate:
                break

        if state.reqs_allocated < state.reqs_to_allocate:
            logging.debug(
                "Only %d/%d reqs of app %s could be allocated",
                state.reqs_allocated.to("reqs").magnitude,
                state.reqs_to_allocate.to("reqs").magnitude,
                app.name,
            )
            self.no_alloc_found = True

    def __create_sol(self) -> Solution:
        """Create a solution.

        Returns:
            Solution to the problem"""
        if self.no_alloc_found:
            # Empty alloc
            alloc = Allocation(
                vms=self.helper.create_empty_vm_alloc(self.vms),
                containers=self.helper.create_empty_container_alloc(self.containers),
            )
            cost = Currency("0 usd")
            status = Status.INFEASIBLE
        else:
            alloc = self.__create_alloc_from_replica_info()
            cost = self.__compute_cost(alloc)
            status = Status.INTEGER_FEASIBLE

        solving_stats = SolvingStats(
            frac_gap=0,
            max_seconds=0,
            lower_bound=0,
            creation_time=self.creation_time,
            solving_time=time.perf_counter() - self.start_solving,
            status=status,
        )

        return Solution(
            problem=self.problem,
            alloc=alloc,
            cost=cost,
            solving_stats=solving_stats,
        )

    def __create_alloc_from_replica_info(self) -> Allocation:
        """Create an allocation from the replica info."""
        vm_alloc = self.helper.create_empty_vm_alloc(self.vms)
        container_alloc = self.helper.create_empty_container_alloc(self.containers)

        for vm in self.ordered_vms:
            for replica_info in self.replica_info[vm]:
                cc = replica_info.container_class
                num_replicas = replica_info.num_replicas
                if num_replicas > 0:
                    vm_alloc[vm] = True
                    container_name = Container(cc, vm).name()
                    container = self.containers[container_name]
                    container_alloc[container] = num_replicas

        return Allocation(vm_alloc, container_alloc)

    def __compute_cost(self, alloc: Allocation) -> Currency:
        """Compute the cost of an allocation.

        Args:
            alloc: allocation

        Returns:
            Cost of the allocation"""
        cost = Currency("0 usd")

        for vm in self.vms.values():
            if alloc.vms[vm]:
                cost += vm.ic.price * self.problem.sched_time_size

        return cost

    def allocate_replicas_in_vm(
        self, app: App, vm: Vm, state: FirstFitAllocatorState
    ) -> None:
        """Allocate replicas of an app in a VM.

        Args:
            app: app to allocate
            vm: vm where to allocate
            state: state of the allocator"""
        replica_info_for_app = [
            replica_info
            for replica_info in self.replica_info[vm]
            if replica_info.container_class.app == app
        ]
        for replica_info in replica_info_for_app:
            logging.debug(
                "  Allocating replicas of %s in vm %s",
                replica_info.container_class.name,
                vm.name(),
            )
            if state.reqs_allocated >= state.reqs_to_allocate:
                break

            while replica_info.num_replicas < replica_info.container_class.limit:
                cc = replica_info.container_class
                if self.cc_fits_in_vm(cc, vm):
                    replica_info.num_replicas += 1

                    ic = vm.ic
                    perfs = self.problem.system.perfs
                    slot_size = self.problem.sched_time_size
                    perf = perfs[(ic, cc)] * slot_size
                    state.reqs_allocated += perf
                else:
                    break

                if state.reqs_allocated >= state.reqs_to_allocate:
                    break

    def cc_fits_in_vm(self, cc: ContainerClass, vm: Vm) -> bool:
        """Check if a container class fits in a VM taking into account the
        already allocated container classes.

        Args:
            cc: container class
            vm: VM

        Returns:
            True if the container class fits in the VM, False otherwise"""
        cores = ComputationalUnits("0 cores")
        mem = Storage("0 bytes")
        for rep_info in self.replica_info[vm]:
            cores += rep_info.num_replicas * rep_info.container_class.cores
            mem += rep_info.num_replicas * rep_info.container_class.mem

        return cores + cc.cores <= vm.ic.cores and mem + cc.mem <= vm.ic.mem
