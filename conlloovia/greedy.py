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

    def __init__(self, vm_alloc, container_alloc) -> None:
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

        # Precompute the cheapest instance class and the VMs of that IC
        self.cheapest_ic = self.helper.compute_cheapest_ic()
        self.cheapest_vms = [
            vm for vm in self.vms.values() if vm.ic == self.cheapest_ic
        ]

        # Precompute the smallest container class for each app
        self.smallest_ccs_per_app: Dict[App, ContainerClass] = {}
        for app in problem.system.apps:
            ccs_for_app = [cc for cc in problem.system.ccs if cc.app == app]
            self.smallest_ccs_per_app[app] = min(ccs_for_app, key=lambda cc: cc.cores)

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

    def _allocate_ccs_for_app(self, app: App, state: GreedyAllocatorState) -> None:
        """Allocates the container classes for the given app, updating the
        state."""
        cc = self.smallest_ccs_per_app[app]
        num_ccs = self._compute_num_ccs_for_app(app)

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

        for _ in range(num_ccs):
            self._allocate_cc(cc, state)
            if state.no_alloc_found:
                return

    def _allocate_cc(self, cc: ContainerClass, state: GreedyAllocatorState) -> None:
        """Allocates a container class replica, updating the state."""
        new_cores = state.cores + cc.cores
        new_mem = state.mem + cc.mem

        if self._is_new_vm_needed(state.current_vm, new_cores, new_mem):
            if state.num_vms == len(self.cheapest_vms):
                logging.info("  Not enough VMs")
                state.no_alloc_found = True
                return

            self._create_new_vm(state)

        assert state.current_vm is not None  # for mypy

        state.cores = state.cores + cc.cores
        state.mem = state.mem + cc.mem
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
        if state.container_alloc[container] > cc.limit:
            logging.info(
                "  Not enough containers of class %s in VM %s-%i",
                cc.name,
                state.current_vm.ic.name,
                state.current_vm.num,
            )
            state.no_alloc_found = True
            return

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

    def _compute_num_ccs_for_app(self, app) -> int:
        """Computes the number of CCs needed for the given app according to
        its workload."""
        cc = self.smallest_ccs_per_app[app]
        cc_perf = self.problem.system.perfs[self.cheapest_ic, cc]
        cc_reqs_in_sched_ts = cc_perf * self.problem.sched_time_size
        wl_reqs = self.problem.workloads[app].num_reqs
        return math.ceil(wl_reqs / cc_reqs_in_sched_ts)

    def _is_new_vm_needed(
        self, current_vm: Optional[Vm], new_cores: ComputationalUnits, new_mem: Storage
    ) -> bool:
        """Checks if a new VM is needed. This will happen if the current VM is
        None (i.e., we are allocating the first CC) or if the new number of
        cores or memory exceeds the number of cores or memory of the current
        VM."""
        return (
            current_vm is None
            or new_cores > current_vm.ic.cores
            or new_mem > current_vm.ic.mem
        )