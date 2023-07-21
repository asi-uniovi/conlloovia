"""This module defines FirstFitAllocator, which is a simple allocator that
works like this:

- Create a list of VMs ordered by decreasing number of cores or increasing price per core.
  This is the list of machines that we will try to fit the containers into.
- For each VM, create a list of containers ordered by increasing size of cores and memory.
- Order the apps by increasing size of their smallest container.
- For each app, allocate the containers in the order of the list of VMs and containers
  inside each VM. If we run out of VMs, the allocation is not feasible.
- After a feasible allocation is found, we try to improve it by moving the containers in
  the last VM to the cheapest VM that can fit them and supports the workload.
"""
from enum import Enum
from dataclasses import dataclass
import logging
import time

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


class FirstFitIcOrdering(Enum):
    """Ordering of the Instance Classes in FirstFitAllocator2."""

    CORE_DESCENDING = 1
    PRICE_ASCENDING = 2


class FirstFitAllocator:
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
        if not self.used_vms:
            return Currency("0 usd")

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
