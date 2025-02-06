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
from functools import lru_cache
import logging
import time

from cloudmodel.unified.units import (
    Currency,
    ComputationalUnits,
    Storage,
    Requests,
    RequestsPerTime,
)


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
            problem: problem to solve
            ordering: ordering of the instance classes. It can be CORE_DESCENDING or
              PRICE_ASCENDING"""
        self.problem = problem
        self.ordering = ordering

        start_creation = time.perf_counter()

        # Create list of ICs ordered according to the ordering
        if ordering == FirstFitIcOrdering.CORE_DESCENDING:
            self.sorted_ics = sorted(
                problem.system.ics,
                key=lambda ic: (ic.cores, ic.mem, ic.price.to("usd/h") / ic.cores),
                reverse=True,
            )
        elif ordering == FirstFitIcOrdering.PRICE_ASCENDING:
            self.sorted_ics = sorted(
                problem.system.ics,
                key=lambda ic: (
                    ic.price.to("usd/h") / ic.cores,
                    ic.cores,
                    ic.mem,
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
        self.sorted_apps_ccs_list: list[tuple[App, tuple[ContainerClass, ...]]] = (
            sorted(
                sorted_apps_ccs.items(),
                key=lambda app_tuple: (app_tuple[1][0].cores, app_tuple[1][0].mem),
                reverse=True,
            )
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
        """Solve the problem using this algorithm:

        For each IC [sorted by cores asc, mem asc and price desc]:
            For each app:
                For each container of the app [sorted by cores asc and mem asc]:
                    While perf(app) < workload(app):
                        If the CC can be executed on the IC:
                            Create a container and try to assign it to the current VM
                            If it cannot be assigned (VM full):
                                Create a new VM  // return to the "while" and it will be assigned
                        Else:
                            Go to the next CC of the app

        For each VM:
            For each IC [sorted by price]:
                If price(IC) > price(VM.IC):
                    break // cannot be improved
                Try to replace the VM with a VM of the IC
                If successful:
                    break // it has been improved
        """
        self.start_solving = time.perf_counter()

        # Copy the list of apps and container classes because we will modify it,
        # removing the apps that are already allocated
        non_allocated_apps_ccs = self.sorted_apps_ccs_list[:]
        for ic in self.sorted_ics:
            if not non_allocated_apps_ccs:
                logging.debug("All apps have been allocated")
                break

            logging.debug("Trying to allocate IC %s", ic.name)
            for app, ccs in non_allocated_apps_ccs[:]:
                for cc in ccs:
                    logging.debug(
                        "    Trying to allocate app %s with cc %s", app.name, cc.name
                    )
                    logging.debug(
                        "        Perf remaining: %s",
                        self.__wl_in_period_for_app(app) - self.current_perf[app],
                    )
                    prev_cc = None
                    while self.current_perf[app] < self.__wl_in_period_for_app(app):
                        if not self.ic_can_run_cc(ic, cc):
                            break  # This container class cannot be executed in this IC

                        container_allocated = self.__try_allocate_container(cc, prev_cc)
                        prev_cc = cc

                        if not container_allocated:
                            # Add new VM if possible
                            logging.debug(
                                "No VMs available for app %s with cc %s. Allocating a new VM of IC %s",
                                app.name,
                                cc.name,
                                ic.name,
                            )
                            vm_allocated = self.__try_allocate_vm(cc)
                            if not vm_allocated:
                                logging.debug(
                                    "No more VMs can be allocated for app %s", app.name
                                )
                                return self.__create_aborted_solution()

                    # Break out of the for loop of ccs if the required performance is reached
                    if self.current_perf[app] >= self.__wl_in_period_for_app(app):
                        logging.debug("App %s has been allocated", app.name)
                        # Remove all (app, ccs) tuples with this app so that we don't
                        # try to allocate it again in other ICs
                        non_allocated_apps_ccs = [
                            (app2, ccs2)
                            for app2, ccs2 in non_allocated_apps_ccs
                            if app2 != app
                        ]
                        break

        logging.info(
            "All ICs have been allocated. Number of VMs: %d", len(self.used_vms)
        )
        self.__remove_unused_vms()  # There should not be unused VMs, but just in case
        self.__optimize_vms()

        return self.__create_solution()

    def ic_can_run_cc(self, ic, cc):
        return (
            (ic, cc) in self.problem.system.perfs
            and cc.cores <= ic.cores
            and cc.mem <= ic.mem
        )

    def __optimize_vms(self) -> None:
        """Optimize the allocation by trying to fit the containers of each VM in a cheaper
        VM that can support the workload. For each IC, it only checks if the new IC gives
        more performance than the current one. This doesn't check if the whole allocation
        is feasible with the new IC. For that, see __deep_optimize_vms."""
        logging.info(
            "Optimizing the allocation by trying to replace VMs with cheaper VMs"
        )

        sorted_by_price_ics = self.get_ics_sorted_by_price()

        # Iterate over a copy of the list of used VMs because we will modify it
        for vm_info in self.used_vms[:]:
            self.__optimize_vm(sorted_by_price_ics, vm_info)

        # Check that the final allocation is feasible. It should be, but just in case
        # there is a bug in the code
        if not self.__is_feasible_alloc():
            logging.error("The final allocation is not feasible")
            raise ValueError("The final allocation is not feasible")

    def __optimize_vm(self, sorted_by_price_ics, vm_info):
        logging.debug("Optimizing VM %s", vm_info.vm.name())

        cores_used, mem_used = self.compute_used_resources(vm_info)

        for ic in sorted_by_price_ics:
            if ic.price > vm_info.vm.ic.price:
                logging.debug(
                    "Price of IC %s is higher than the price of VM %s. Cannot be improved",
                    ic.name,
                    vm_info.vm.name(),
                )
                return  # Cannot be improved

            if ic == vm_info.vm.ic:
                continue  # If it is the same IC, we don't need to try to replace it

            if ic.cores < cores_used or ic.mem < mem_used:
                continue  # This IC cannot fit the containers of the VM, try another one

            if not self.__is_perf_improved(vm_info, ic):
                continue  # The performance is not improved, try another IC

            self.__replace_vm(vm_info, ic)
            logging.debug(
                "    VM %s has been replaced with IC %s", vm_info.vm.name(), ic.name
            )
            return  # It has been improved

        logging.debug(
            "    No IC can replace VM %s. It cannot be improved",
            vm_info.vm.name(),
        )

    def get_ics_sorted_by_price(self):
        """Get a list of the ICs ordered by price in ascending order, then cores in
        descending order and memory in descending order."""
        return sorted(
            self.problem.system.ics,
            key=lambda ic: (
                ic.price.to("usd/h"),
                -ic.cores.to("core"),
                -ic.mem.to("GB"),
            ),
        )

    def __replace_vm(self, vm_info, ic):
        """Replace a VM with a new VM of a different IC. It doesn't check if the new IC
        gives more performance than the current one. It only replaces the VM and moves
        the containers. It doesn't check if the whole allocation is feasible with the new
        IC."""
        new_vm = Vm(ic, self.num_vms)
        self.num_vms += 1
        new_vm_info = VmInfo(new_vm, [])

        # Move the containers from the old VM to the new VM
        self.__move_containers(vm_info, new_vm_info)

        # Remove the old one and append the new one
        self.used_vms.remove(vm_info)
        self.used_vms.append(new_vm_info)

    def __is_perf_improved(self, vm_info, new_ic):
        """Check if the performance is improved by replacing a VM with a new VM of a
        different IC. It doesn't check if the whole allocation is feasible with the new
        IC, just if the performance in rps is higher with the new IC."""
        # Check that, for each app, the performance in rps is higher with the new IC
        rps_old = {app: RequestsPerTime("0 req/s") for app in self.problem.system.apps}
        rps_new = {app: RequestsPerTime("0 req/s") for app in self.problem.system.apps}
        old_ic = vm_info.vm.ic
        for ri_info in vm_info.ri_list:
            cc = ri_info.cc
            app = cc.app
            if not (new_ic, cc) in self.problem.system.perfs:
                return False  # This cc cannot be executed in this ic
            rps_old[app] += self.problem.system.perfs[(old_ic, cc)].to("req/s")
            rps_new[app] += self.problem.system.perfs[(new_ic, cc)].to("req/s")

        for app in self.problem.system.apps:
            if rps_new[app] < rps_old[app]:
                return False

        return True

    def __deep_optimize_vms(self) -> None:
        """Optimize the allocation by trying to fit the containers of each VM in a cheaper
        VM that can support the workload. It's called deep because it tries a new IC and
        then checks that the whole allocation is feasible, not just if the new IC gives
        more performance than the current one. If only the latter is checked, there could
        be situations where the new IC doesn't give more performance but the whole
        allocation is feasible with it."""

        logging.info(
            "Optimizing the allocation by trying to replace VMs with cheaper VMs"
        )

        sorted_by_price_ics = self.get_ics_sorted_by_price()

        # Create a dictionary indictaing for each IC if it can be optimized and how, i.e.,
        # with which IC it can be replaced. It's initially empty
        best_ic_to_replace = {}

        # Iterate over a copy of the list of used VMs because we will modify it
        for vm_info in self.used_vms[:]:
            logging.info("Optimizing VM %s", vm_info.vm.name())

            # Compute the number of cores and memory used in the VM. It is computed here
            # instead of in __try_replace_vm to avoid recomputing it for each IC
            cores_used, mem_used = self.compute_used_resources(vm_info)

            for ic in sorted_by_price_ics:
                # If it is the same IC, we don't need to try to replace it
                if ic == vm_info.vm.ic:
                    continue

                if ic.price > vm_info.vm.ic.price:
                    logging.debug(
                        "Price of IC %s is higher than the price of VM %s. Cannot be improved",
                        ic.name,
                        vm_info.vm.name(),
                    )
                    best_ic_to_replace[vm_info.vm.ic] = None
                    break  # Cannot be improved

                if self.__try_replace_vm(vm_info, ic, cores_used, mem_used):
                    logging.debug(
                        "    VM %s has been replaced with IC %s",
                        vm_info.vm.name(),
                        ic.name,
                    )
                    best_ic_to_replace[vm_info.vm.ic] = ic
                    break  # It has been improved, move to the next VM
            else:
                logging.debug(
                    "    No IC can replace VM %s. It cannot be improved",
                    vm_info.vm.name(),
                )
                best_ic_to_replace[vm_info.vm.ic] = None

    def __try_replace_vm(
        self,
        vm_info: VmInfo,
        ic: InstanceClass,
        cores_used: ComputationalUnits,
        mem_used: Storage,
    ) -> bool:
        """Try to replace a VM with a VM of a cheaper IC that can support the workload.
        Returns True if the VM has been replaced, False otherwise. Modifies the
        allocation. The parameters cores_used and mem_used are the number of cores and
        memory used in the VM, respectively."""
        logging.debug(
            "      Trying to replace VM %s with IC %s", vm_info.vm.name(), ic.name
        )

        if ic.cores < cores_used or ic.mem < mem_used:
            return False  # The IC cannot fit the containers of the VM

        # Create a new VM with this IC
        logging.debug("      Creating new VM with IC %s", ic.name)
        new_vm = Vm(ic, self.num_vms)
        self.num_vms += 1
        new_vm_info = VmInfo(new_vm, [])

        # Remove the old one and append the new one
        self.used_vms.remove(vm_info)
        self.used_vms.append(new_vm_info)

        # Move the containers from the old VM to the new VM
        self.__move_containers(vm_info, new_vm_info)

        # Check that the allocation is still feasible
        if self.__is_feasible_alloc():
            return True  # It is, we are done

        # It's not feasible, so undo the changes
        logging.debug(
            "      The allocation is not feasible after moving the containers "
            "Undoing the changes..."
        )
        self.__move_containers(new_vm_info, vm_info)
        self.used_vms.remove(new_vm_info)
        self.used_vms.append(vm_info)

        return False

    @staticmethod
    def compute_used_resources(vm_info) -> tuple[ComputationalUnits, Storage]:
        """Compute the number of cores and memory used in a VM."""
        cores_used = ComputationalUnits("0 cores")
        mem_used = Storage("0 GB")
        for ri_src in vm_info.ri_list:
            cores_used += ri_src.num_replicas * ri_src.cc.cores
            mem_used += ri_src.num_replicas * ri_src.cc.mem
        return cores_used, mem_used

    def __wl_in_period_for_app(self, app) -> Requests:
        """Get the workload in the period for an app."""
        return self.problem.workloads[app].num_reqs

    def __try_allocate_container(
        self, cc: ContainerClass, prev_cc: ContainerClass
    ) -> bool:
        """Try to allocate a container of container class cc in the current VMs.
        If the previous container class prev_cc is given and is equal to cc, the
        allocation will start in the last VM, because previous VMs are already checked for
        that cc.

        Returns:
            True if the container is allocated, False otherwise"""
        if prev_cc == cc:
            vms_to_check = self.used_vms[-1:]
        else:
            vms_to_check = self.used_vms

        for vm_info in vms_to_check:
            vm = vm_info.vm
            ri_list = vm_info.ri_list
            logging.debug(
                "    Trying to allocate container %s in VM %s. Remaining perf: %s",
                cc.name,
                vm.name(),
                self.__wl_in_period_for_app(cc.app) - self.current_perf[cc.app],
            )
            if (vm.ic, cc) in self.problem.system.perfs and vm_info.cc_fits(cc):
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

    def __try_allocate_vm(self, cc: ContainerClass) -> bool:
        """Try to allocate a new VM that can run (has perf info) for cc.

        Returns:
            True if a VM can be allocated, False otherwise"""
        if len(self.sorted_ics) == 0:
            logging.debug("No more ICs available")
            return False

        # Create a new VM with the first IC that can run cc, i.e., the biggest remaining
        # IC that has perf info for cc
        for ic in self.sorted_ics:
            if self.ic_can_run_cc(ic, cc):
                break
        else:
            logging.debug("No IC can run cc %s", cc.name)
            return False

        logging.debug("Creating VM with IC %s", ic.name)
        vm = Vm(ic, self.num_vms)
        self.num_vms += 1
        self.used_vms.append(VmInfo(vm, []))

        return True

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

    def __create_aborted_solution(self) -> Solution:
        """Creates an aborted solution."""
        solving_stats = SolvingStats(
            frac_gap=0,
            max_seconds=0,
            lower_bound=0,
            creation_time=self.creation_time,
            solving_time=time.perf_counter() - self.start_solving,
            status=Status.ABORTED,
        )

        return Solution(
            problem=self.problem,
            alloc=Allocation(*self.__create_empty_allocs()),
            cost=Currency("0 usd"),
            solving_stats=solving_stats,
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
                if not (ic, cc) in self.problem.system.perfs:
                    # This container class cannot be executed in this instance class
                    return False
                reqs_provided_period_1_rep = (
                    self.problem.system.perfs[(ic, cc)] * self.problem.sched_time_size
                )
                provided_reqs[app] += ri_info.num_replicas * reqs_provided_period_1_rep

        # Check that the performance is enough for each app
        for app in self.problem.system.apps:
            extra_reqs = provided_reqs[app] - self.problem.workloads[app].num_reqs
            if extra_reqs < Requests("-1e-9 req"):  # Allow a small error
                return False

        return True

    def __remove_unused_vms(self) -> None:
        """Remove the VMs that are not used."""
        self.used_vms = [vm_info for vm_info in self.used_vms if vm_info.ri_list]
