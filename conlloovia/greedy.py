"""This module defines GreedyAllocator, which is a simple greedy allocator that
chooses the cheapest instance class for each app in terms of performance per
dollar."""

import math
import logging
import time
from typing import Dict, Optional

from cloudmodel.unified.units import Currency, ComputationalUnits, Storage

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
from .problem_helper import ProblemHelper


class GreedyAllocatorState:
    """State of the GreedyAllocator while building a solution."""

    def __init__(
        self, vm_alloc: dict[Vm, bool], container_alloc: dict[Container, int]
    ) -> None:
        """Constructor. It assumes that the solving time starts when an object
        of this class is created. The vm_alloc and the container_alloc contain
        the allocation as it is being built. The current_vm is the VM that is
        is currently being filled with containers.

        If the final state is reached an no allocation is found, the
        no_alloc_found variable will be set to True."""
        self.start_solving = time.perf_counter()
        self.vm_alloc = vm_alloc
        self.container_alloc = container_alloc
        self.num_vms = 0  # Number of VMs used
        self.current_vm: Optional[Vm] = None
        self.cores = ComputationalUnits(
            "0 cores"
        )  # Number of used cores in the current VM
        self.mem = Storage("0 bytes")  # Memory used in the current VM
        self.cost = Currency("0 usd")  # Cost of the allocation so far
        self.no_alloc_found = False  # If no allocation is found, it will be changed
        self.requests_served = 0  # Number of requests served by the allocation


class GreedyAllocator:
    """Greedy allocator that only uses one instance class, and one container
    class for each application. It selects the cheapest instance class (in terms
    of cores per dollar) and the smallest container class (in terms of cores)
    for each application."""

    def __init__(self, problem: Problem) -> None:
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem

        self.helper = ProblemHelper(problem)

        start_creation = time.perf_counter()

        self.vms: dict[str, Vm] = self.helper.create_vms_dict()
        self.containers: dict[str, Container] = self.helper.create_containers_dict(
            self.vms
        )

        # Precompute the smallest container class for each app
        self.smallest_ccs_per_app: Dict[App, ContainerClass] = {}
        for app in problem.system.apps:
            ccs_for_app = [cc for cc in problem.system.ccs if cc.app == app]
            self.smallest_ccs_per_app[app] = min(ccs_for_app, key=lambda cc: cc.cores)

        # Precompute the cheapest instance class that can run the smallest container
        # classes and the VMs of that IC
        self.ordered_ics = self.helper.get_ics_ordered()
        self.cheapest_ic = self._compute_cheapest_ic()
        self.cheapest_vms = self._regenerate_cheapest_vms()

        end_creation = time.perf_counter()
        self.creation_time = end_creation - start_creation

    def solve(self) -> Solution:
        """Solves the problem using a Greedy algorithm.

        It computes the number of VMs needed in total. It starts assigning CCs
        and when the number of cores (or memory) of the VM exceeds the number of
        cores (or memory) of the cheapest instance class, it creates a new VM.
        It creates the allocation and computes the cost at the same time. The
        current state of the greedy allocator is stored in the state variable.
        The final state will be used to create the solution.

        Returns:
            solution to the problem
        """
        state = GreedyAllocatorState(
            vm_alloc=self.helper.create_empty_vm_alloc(self.vms),
            container_alloc=self.helper.create_empty_container_alloc(self.containers),
        )

        for app in self.problem.system.apps:
            self._allocate_ccs_for_app(app, state)

            if state.no_alloc_found:
                break

        return self._create_sol_from_state(state)

    def _regenerate_cheapest_vms(self) -> list[Vm]:
        """Regenerates the cheapest VMs.

        Returns:
            list of VMs of the cheapest instance class
        """
        return [vm for vm in self.vms.values() if vm.ic == self.cheapest_ic]

    def _allocate_ccs_for_app(self, app: App, state: GreedyAllocatorState) -> None:
        """Allocates the container classes for the given app, updating the
        state."""
        if self.cheapest_ic is None:
            logging.info(
                "No instance class that can allocate all container classes found"
            )
            state.no_alloc_found = True
            return

        cc = self.smallest_ccs_per_app[app]
        state.requests_served = 0
        num_ccs = self._compute_num_ccs_for_app(
            app, self.problem.workloads[app].num_reqs
        )

        logging.info("Allocating %i %s CCs for app %s", num_ccs, cc.name, app.name)

        if cc.cores > self.cheapest_ic.cores:
            logging.info(
                "  Not enough cores for containers of app %s in the cheapest VM",
                app.name,
            )
            state.no_alloc_found = True
            return

        if cc.mem > self.cheapest_ic.mem:
            logging.info(
                "  Not enough memory for containers of app %s in the cheapest VM",
                app.name,
            )
            state.no_alloc_found = True
            return

        continue_ = True
        while continue_:
            for _ in range(num_ccs):
                self._allocate_cc(cc, state)
                if state.no_alloc_found:
                    logging.info("  Trying next Instance Class")
                    continue_ = self._try_next_ic(state)
                    break
            else:  # no break
                continue_ = False
            if continue_:
                num_ccs = self._compute_num_ccs_for_app(
                    app, self.problem.workloads[app].num_reqs - state.requests_served
                )

    def _try_next_ic(self, state: GreedyAllocatorState) -> bool:
        self.ordered_ics.pop(0)
        if not self.ordered_ics:
            return False
        self.cheapest_ic = self._compute_cheapest_ic()
        self.cheapest_vms += self._regenerate_cheapest_vms()
        state.no_alloc_found = False
        return True

    def _allocate_cc(self, cc: ContainerClass, state: GreedyAllocatorState) -> None:
        """Allocates a container class replica, updating the state."""
        while self._is_new_vm_needed(cc, state):
            if state.num_vms == len(self.cheapest_vms):
                logging.info("  Not enough VMs of this Instance Class")
                state.no_alloc_found = True
                return

            self._create_new_vm(state)

        assert state.current_vm is not None  # for mypy

        state.cores = state.cores + cc.cores
        state.mem = state.mem + cc.mem
        state.requests_served += self._compute_reqs_served_in_ts(
            state.current_vm.ic, cc
        )
        logging.info(
            "    Using %s of VM %i (total of %s/%s, %s/%s)",
            cc.cores,
            state.num_vms - 1,
            state.cores,
            state.current_vm.ic.cores,
            state.mem.to("GiB"),
            state.current_vm.ic.mem,
        )

        container = self.containers[
            f"{state.current_vm.ic.name}-{state.current_vm.num}-{cc.name}"
        ]
        state.container_alloc[container] += 1

    def _create_new_vm(self, state: GreedyAllocatorState) -> None:
        """Creates a new VM, updating the state."""
        state.current_vm = self.cheapest_vms[state.num_vms]
        state.cost += state.current_vm.ic.price * self.problem.sched_time_size
        state.vm_alloc[state.current_vm] = True

        state.num_vms += 1
        state.cores = ComputationalUnits("0 cores")
        state.mem = Storage("0 bytes")

        logging.info(
            "  Using %i/%i VMs (%s)",
            state.num_vms,
            len(self.cheapest_vms),
            state.cost,
        )

    def _create_sol_from_state(self, state: GreedyAllocatorState) -> Solution:
        """Creates the solution from the state, which is supposed to be the
        state after the allocation has been created."""
        if state.no_alloc_found:
            # Empty alloc
            alloc = Allocation(
                vms=self.helper.create_empty_vm_alloc(self.vms),
                containers=self.helper.create_empty_container_alloc(self.containers),
            )
            status = Status.INFEASIBLE
            cost = Currency("0 usd")
        else:
            # Final alloc
            alloc = Allocation(
                vms=state.vm_alloc,
                containers=state.container_alloc,
            )
            status = Status.INTEGER_FEASIBLE
            cost = state.cost

        solving_stats = SolvingStats(
            frac_gap=0,
            max_seconds=0,
            lower_bound=0,
            creation_time=self.creation_time,
            solving_time=time.perf_counter() - state.start_solving,
            status=status,
        )

        sol = Solution(
            problem=self.problem,
            alloc=alloc,
            cost=cost,
            solving_stats=solving_stats,
        )

        return sol

    def _compute_cheapest_ic(self) -> Optional[InstanceClass]:
        """Returns the cheapest instance class, in terms of cores per dollar,
        that can allocate the smallest container classes. If there are several,
        select the one with the smallest number of cores. If none can allocate
        the smallest container classes, return None."""
        # Make a copy of ordered_ics because we will modify it while iterating
        # over it
        ordered_ics_copy = self.ordered_ics.copy()

        for ic in ordered_ics_copy:
            can_allocate_all_smallest_ccs = all(
                cc.cores <= ic.cores
                and cc.mem <= ic.mem
                and (ic, cc) in self.problem.system.perfs
                for cc in self.smallest_ccs_per_app.values()
            )
            if can_allocate_all_smallest_ccs:
                return ic

            # This ic cannot allocate the smallest cc for some app, so we remove
            # it from the list and try the next one
            self.ordered_ics.remove(ic)

        # No ic can allocate the smallest cc for all apps
        return None

    def _compute_reqs_served_in_ts(self, ic: InstanceClass, cc: ContainerClass) -> int:
        """Computes the number of requests served in a scheduling time size
        by the given app, instance class and container class."""
        cc_perf = self.problem.system.perfs[ic, cc]
        return cc_perf * self.problem.sched_time_size

    def _compute_num_ccs_for_app(self, app: App, reqs_to_serve: float) -> int:
        """Computes the number of CCs needed for the given app according to
        its workload."""
        cc = self.smallest_ccs_per_app[app]
        cc_reqs_in_sched_ts = self._compute_reqs_served_in_ts(self.cheapest_ic, cc)
        return math.ceil(reqs_to_serve / cc_reqs_in_sched_ts)

    def _is_new_vm_needed(
        self,
        cc: ContainerClass,
        state: GreedyAllocatorState,
    ) -> bool:
        """Checks if a new VM is needed. This will happen if the current VM is
        None (i.e., we are allocating the first CC), or the container class
        cannot be allocated in the VM, which happens when there is no
        performance information for the tuple (vm.ic, cc), or if the new number
        of cores or memory exceeds the number of cores or memory of the current
        VM, or the maximum number of containers for this container class has
        been reached in this VM."""
        if state.current_vm is None:
            return True

        if (state.current_vm.ic, cc) not in self.problem.system.perfs:
            return True

        new_cores = state.cores + cc.cores
        new_mem = state.mem + cc.mem
        container = Container(cc, state.current_vm)
        n_replicas = state.container_alloc[container]
        return (
            new_cores > state.current_vm.ic.cores
            or new_mem > state.current_vm.ic.mem
            or n_replicas >= cc.limit
        )
