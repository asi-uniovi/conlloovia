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

There's also a second implementation of this allocator called
FirstFitAllocator2. While the first version creates and empty allocation and
then fills it, the second version creates the allocation while it's filling it.
"""
from enum import Enum
from dataclasses import dataclass
import logging
import time
from typing import Optional

from cloudmodel.unified.units import Currency, ComputationalUnits, Storage, Requests

from conlloovia import problem_helper
from .model import (
    Problem,
    InstanceClass,
    ContainerClass,
    App,
    Allocation,
    Vm,
    Container,
    Solution,
    Status,
    SolvingStats,
)


@dataclass
class ReplicaInfo:
    """Stores information about the number of replicas of a container class in
    a VM."""

    cc: ContainerClass
    num_replicas: int


@dataclass
class VmInfo:
    """Stores information about the number of replicas of each container class
    in a VM and the number of cores and mem used."""

    vm: Vm
    ri_list: list[ReplicaInfo]
    cores: ComputationalUnits = ComputationalUnits("0 cores")
    mem: Storage = Storage("0 GB")

    def cc_fits(self, cc: ContainerClass) -> bool:
        """Check if a container class fits in the VM."""
        return (
            self.cores + cc.cores <= self.vm.ic.cores
            and self.mem + cc.mem <= self.vm.ic.mem
        )


@dataclass
class FirstFitAllocatorState:
    """State of the FirstFitAllocator while building a solution."""

    # Number of allocated requests of the current app
    reqs_allocated = Requests("0 req")

    # Number of requests to allocate for the current app
    reqs_to_allocate = Requests("0 req")


class FirstFitAllocator:
    """First fit allocator."""

    def __init__(self, problem: Problem) -> None:
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem

        start_creation = time.perf_counter()

        self.vms: dict[str, Vm] = problem_helper.create_vms_dict(problem.system.ics)
        self.containers: dict[str, Container] = problem_helper.create_containers_dict(
            problem.system.ccs, self.vms
        )

        self.ordered_vms: list[Vm] = problem_helper.get_vms_ordered_by_cores_desc(
            self.vms
        )

        # Create a dictionary where the index is a VM and the value is a list of
        # ReplicaInfo objects, ordered by increasing number of cores and memory.
        ordered_ccs = problem_helper.get_ccs_ordered_by_cores_and_mem_asc(
            problem.system.ccs
        )
        self.replica_info: dict[Vm, list[ReplicaInfo]] = {}
        for vm in self.ordered_vms:
            replica_info_list = []
            for cc in ordered_ccs:
                replica_info_list.append(ReplicaInfo(cc, 0))
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

                print(f"  {rep_info.cc.name} -> " f"{rep_info.num_replicas} replicas")

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
        for app in problem_helper.get_apps_ordered_by_container_size_asc(
            self.problem.system.ccs
        ):
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
            cores_used += ri_src.num_replicas * ri_src.cc.cores
            mem_used += ri_src.num_replicas * ri_src.cc.mem

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
                if rep_info.cc.app == app:
                    perf_app += (
                        rep_info.num_replicas
                        * perfs[vm.ic, rep_info.cc]
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
            if ri_dst.cc != ri_src.cc:
                continue  # Different container class

            logging.debug(
                "Moving %d replicas of %s from %s to %s",
                ri_src.num_replicas,
                ri_src.cc.name,
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
                vms=problem_helper.create_empty_vm_alloc(self.vms),
                containers=problem_helper.create_empty_container_alloc(self.containers),
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
        vm_alloc = problem_helper.create_empty_vm_alloc(self.vms)
        container_alloc = problem_helper.create_empty_container_alloc(self.containers)

        for vm in self.ordered_vms:
            for replica_info in self.replica_info[vm]:
                cc = replica_info.cc
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
            if replica_info.cc.app == app
        ]
        for replica_info in replica_info_for_app:
            logging.debug(
                "  Allocating replicas of %s in vm %s",
                replica_info.cc.name,
                vm.name(),
            )
            if state.reqs_allocated >= state.reqs_to_allocate:
                break

            while True:
                cc = replica_info.cc
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
            cores += rep_info.num_replicas * rep_info.cc.cores
            mem += rep_info.num_replicas * rep_info.cc.mem

        return cores + cc.cores <= vm.ic.cores and mem + cc.mem <= vm.ic.mem


class FirstFitIcOrdering(Enum):
    """Ordering of the Instance Classes in FirstFitAllocator2."""

    CORE_DESCENDING = 1
    PRICE_ASCENDING = 2


class FirstFitAllocator2:
    """First fit allocator."""

    def __init__(self, problem: Problem, ordering: FirstFitIcOrdering) -> None:
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem
        self.ordering = ordering

        start_creation = time.perf_counter()

        # Create list of ICs ordered according to the ordering
        if ordering == FirstFitIcOrdering.CORE_DESCENDING:
            self.sorted_ics = sorted(
                problem.system.ics,
                key=lambda ic: (ic.cores, ic.mem),
                reverse=True,
            )
        elif ordering == FirstFitIcOrdering.PRICE_ASCENDING:
            self.sorted_ics = sorted(
                problem.system.ics,
                key=lambda ic: (
                    ic.price.to("usd/h") / ic.cores,
                    ic.cores,
                ),
            )
        else:
            raise ValueError(f"Unknown ordering {ordering}")

        sorted_apps_ccs = {}
        for app in self.problem.system.apps:
            ccs = problem_helper.get_ccs_for_app_ordered_by_cores_and_mem_asc(
                problem.system.ccs, app
            )
            sorted_apps_ccs[app] = ccs

        # Transform sorted_apps_ccs dictionary into a list of tuples, where apps
        # are ordered by decreasing number of cores and mem of the first
        # container class, which is the smallest container class for the app.
        # Note that app_tuple[1] is a list of container classes, and
        # app_tuple[1][0] is the first container class, which is the smallest
        # container class for the app.
        self.sorted_apps_ccs_list: list[
            tuple[App, tuple[ContainerClass, ...]]
        ] = sorted(
            sorted_apps_ccs.items(),
            key=lambda app_tuple: (app_tuple[1][0].cores, app_tuple[1][0].mem),
            reverse=True,
        )

        # Current used VMs, initially none
        self.used_vms: list[VmInfo] = []

        # Number of VMs currently used, initially zero
        self.num_vms = 0

        # Current performance in requests for each app, initially zero for all
        # apps
        self.current_perf = {
            app: Requests("0 reqs") for app in self.problem.system.apps
        }

        end_creation = time.perf_counter()
        self.creation_time = end_creation - start_creation

        # Time when the solving starts
        self.start_solving = 0.0

    def solve(self) -> Solution:
        """Solve the problem.

        Returns:
            solution"""
        self.start_solving = time.perf_counter()

        while self.__perf_below_workload():
            # Take the first app and its container classes
            app, ccs = self.sorted_apps_ccs_list[0]

            logging.debug(
                "Trying to allocate app %s with ccs %s", app.name, ccs[0].name
            )

            # Try to allocate the first container of the app in the current VMs
            container_allocated = self.__try_allocate_container(ccs[0])

            if not container_allocated:
                # Add new VM if possible
                logging.debug("No VMs available for app %s", app.name)
                vm_allocated = self.__try_allocate_vm()
                if not vm_allocated:
                    logging.debug("No more VMs can be allocated for app %s", app.name)
                    return self.__create_infeasible_solution()
            else:
                # Check the performance requirement for this app
                if self.current_perf[app] >= self.__wl_in_period_for_app(app):
                    # Remove the app from the list of apps because it is
                    # already allocated
                    self.sorted_apps_ccs_list.pop(0)

        self.__optimize_last_vm()

        return self.__create_solution()

    def __perf_below_workload(self) -> bool:
        """Check if the current performance is below the workload for any app.

        Returns:
            True if the current performance is below the workload for any app,
            False otherwise"""
        for app in self.problem.system.apps:
            if self.current_perf[app] < self.__wl_in_period_for_app(app):
                return True

        return False

    def __wl_in_period_for_app(self, app) -> Requests:
        """Get the workload in the period for an app."""
        return self.problem.workloads[app].num_reqs

    def __try_allocate_container(self, cc: ContainerClass) -> bool:
        """Try to allocate a container of container class cc in the current VMs.

        Returns:
            True if the container is allocated, False otherwise"""
        for vm_info in self.used_vms:
            vm = vm_info.vm
            ri_list = vm_info.ri_list
            logging.debug(
                "    Trying to allocate container %s in VM %s", cc.name, vm.name()
            )
            if vm_info.cc_fits(cc):
                logging.debug(
                    "        Allocated container %s in VM %s", cc.name, vm.name()
                )
                perf = (
                    self.problem.system.perfs[vm.ic, cc] * self.problem.sched_time_size
                )
                self.current_perf[cc.app] += perf

                # Check if there is a replica info for the container class in
                # ri_list
                for ri_info in ri_list:
                    if ri_info.cc == cc:
                        # Increase the number of replicas
                        ri_info.num_replicas += 1
                        vm_info.cores = vm_info.cores + cc.cores
                        vm_info.mem = vm_info.mem + cc.mem
                        return True

                # Create a new replica info with one replica
                new_ri = ReplicaInfo(cc, 1)
                ri_list.append(new_ri)
                vm_info.cores = vm_info.cores + cc.cores
                vm_info.mem = vm_info.mem + cc.mem
                return True

        # Didn't find a VM where the container class fits
        return False

    def __try_allocate_vm(self) -> bool:
        """Try to allocate a new VM.

        Returns:
            True if a VM can be allocated, False otherwise"""
        if len(self.sorted_ics) == 0:
            logging.debug("No more ICs available")
            return False

        # Create a new VM with the first IC, i.e., the biggest remaining IC
        ic = self.sorted_ics[0]
        logging.debug("Creating VM with IC %s", ic.name)
        vm = Vm(ic, self.num_vms)
        self.num_vms += 1
        self.used_vms.append(VmInfo(vm, []))

        # Check that we have not reached the maximum number of VMs of this IC
        if self.__num_vms_of_ic(ic) == ic.limit:
            # No more VMs of this IC can be created, so remove it from the list
            self.sorted_ics.pop(0)

        return True

    def __num_vms_of_ic(self, ic: InstanceClass) -> int:
        """Get the number of VMs used of an instance class."""
        return sum(1 for vm_info in self.used_vms if vm_info.vm.ic == ic)

    def __create_solution(self) -> Solution:
        """Creates a solution with the current state."""
        solving_stats = SolvingStats(
            frac_gap=0,
            max_seconds=0,
            lower_bound=0,
            creation_time=self.creation_time,
            solving_time=time.perf_counter() - self.start_solving,
            status=Status.INTEGER_FEASIBLE,
        )

        return Solution(
            problem=self.problem,
            alloc=self.__create_alloc_from_replica_info(),
            cost=self.__compute_cost(),
            solving_stats=solving_stats,
        )

    def __create_alloc_from_replica_info(self) -> Allocation:
        """Creates an allocation from the current state."""
        vm_alloc, container_alloc = self.__create_empty_allocs()

        for vm_info in self.used_vms:
            for replica_info in vm_info.ri_list:
                cc = replica_info.cc
                num_replicas = replica_info.num_replicas
                if num_replicas > 0:
                    vm_alloc[vm_info.vm] = True
                    container = Container(cc, vm_info.vm)
                    container_alloc[container] = num_replicas

        return Allocation(vm_alloc, container_alloc)

    def __create_empty_allocs(self):
        """Creates empty allocation dictionaries for VMs and containers."""
        vms_dict = problem_helper.create_vms_dict(self.problem.system.ics)
        containers_dict = problem_helper.create_containers_dict(
            self.problem.system.ccs, vms_dict
        )

        vm_alloc = problem_helper.create_empty_vm_alloc(vms_dict)
        container_alloc = problem_helper.create_empty_container_alloc(containers_dict)
        return vm_alloc, container_alloc

    def __compute_cost(self) -> Currency:
        """Compute the cost of the current state."""
        return sum(
            vm_info.vm.ic.price * self.problem.sched_time_size
            for vm_info in self.used_vms
        )

    def __create_infeasible_solution(self) -> Solution:
        """Creates an infeasible solution."""
        solving_stats = SolvingStats(
            frac_gap=0,
            max_seconds=0,
            lower_bound=0,
            creation_time=self.creation_time,
            solving_time=time.perf_counter() - self.start_solving,
            status=Status.INFEASIBLE,
        )

        return Solution(
            problem=self.problem,
            alloc=Allocation(*self.__create_empty_allocs()),
            cost=Currency("0 usd"),
            solving_stats=solving_stats,
        )

    def __optimize_last_vm(self) -> None:
        """Optimize the allocation by trying to fit the containers from the last
        VM into smaller VMs."""
        if len(self.used_vms) == 0:
            # No VM is used
            return

        last_used_vm_info = self.used_vms[-1]

        # Compute the number of cores and memory used in the last VM
        cores_used = ComputationalUnits("0 cores")
        mem_used = Storage("0 GB")
        for ri_src in last_used_vm_info.ri_list:
            cores_used += ri_src.num_replicas * ri_src.cc.cores
            mem_used += ri_src.num_replicas * ri_src.cc.mem

        # We will try to use the cheapest VM that can fit the containers from
        # the last VM while obtaining a feasible solution. First, we order the
        # remaining ICs by price in ascending order
        sorted_ics_price_asc = sorted(self.sorted_ics, key=lambda ic: ic.price)

        for ic in sorted_ics_price_asc:
            if not (ic.cores >= cores_used and ic.mem >= mem_used):
                continue  # This IC cannot fit the containers from the last VM

            # Create a new VM with this IC
            logging.debug("Creating VM with IC %s", ic.name)
            new_vm = Vm(ic, self.num_vms)
            self.num_vms += 1
            new_vm_info = VmInfo(new_vm, [])

            # Remove the old one and append the new one
            self.used_vms.pop()
            self.used_vms.append(new_vm_info)

            # Move the containers from the last VM to the new VM
            self.__move_containers(last_used_vm_info, new_vm_info)

            # Check that the allocation is still feasible
            if self.__is_feasible_alloc():
                return  # It is, we are done

            # It's not feasible, so undo the changes
            logging.debug(
                "The allocation is not feasible after moving the containers "
                "from the last VM. Undoing the changes..."
            )
            self.__move_containers(new_vm_info, last_used_vm_info)
            self.used_vms.pop()
            self.used_vms.append(last_used_vm_info)

        logging.debug(
            "No cheaper VM that can fit the containers from the last VM"
            " while maintaining a feasible allocation."
        )

    def __move_containers(self, src: VmInfo, dst: VmInfo) -> None:
        """Move the containers from one VM to another."""
        for ri_src in src.ri_list:
            dst.ri_list.append(ri_src)
            dst.cores += ri_src.num_replicas * ri_src.cc.cores
            dst.mem += ri_src.num_replicas * ri_src.cc.mem
        src.ri_list = []
        src.cores = ComputationalUnits("0 cores")
        src.mem = Storage("0 GB")

    def __is_feasible_alloc(self) -> bool:
        """Check that the current used VMs and containers allocation give enough
        performance for the workload."""

        # Compute the provided requests for each app
        provided_reqs = {app: Requests("0 reqs") for app in self.problem.system.apps}
        for vm_info in self.used_vms:
            for ri_info in vm_info.ri_list:
                app = ri_info.cc.app
                cc = ri_info.cc
                ic = vm_info.vm.ic
                reqs_provided_period_1_rep = (
                    self.problem.system.perfs[(ic, cc)] * self.problem.sched_time_size
                )
                provided_reqs[app] += ri_info.num_replicas * reqs_provided_period_1_rep

        # Check that the performance is enough for each app
        for app in self.problem.system.apps:
            if provided_reqs[app] < self.problem.workloads[app].num_reqs:
                return False

        return True
