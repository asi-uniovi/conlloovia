"""Tests for `FirstFitAllocator`."""

import unittest

import pytest
from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    Time,
    Requests,
)

from conlloovia.first_fit import (
    FirstFitAllocator,
    FirstFitIcOrdering,
)
from conlloovia.visualization import SolutionPrettyPrinter, ProblemPrettyPrinter
from conlloovia.model import (
    Workload,
    Problem,
    Status,
)

assertions = unittest.TestCase("__init__")


class Test2apps:
    """Tests for the first-fit allocator with 2 apps."""

    def test_2apps_first_fit_cores_descending(self, system_2apps):
        """Test that the first-fit allocator works with 2 apps using core
        descending ordering."""
        system = system_2apps

        app0 = system.apps[0]
        app1 = system.apps[1]
        workloads = {
            app0: Workload(
                num_reqs=Requests("5 req"), time_slot_size=Time("s"), app=app0
            ),
            app1: Workload(
                num_reqs=Requests("10 req"), time_slot_size=Time("s"), app=app1
            ),
        }
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = FirstFitAllocator(problem, ordering=FirstFitIcOrdering.CORE_DESCENDING)
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        assert sol.cost == Currency("1.6/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 2

    def test_2apps_first_fit_price_ascending(self, system_2apps):
        """Test that the first-fit allocator v2 works with 2 apps using price
        ascending ordering."""
        system = system_2apps

        app0 = system.apps[0]
        app1 = system.apps[1]
        workloads = {
            app0: Workload(
                num_reqs=Requests("5 req"), time_slot_size=Time("s"), app=app0
            ),
            app1: Workload(
                num_reqs=Requests("10 req"), time_slot_size=Time("s"), app=app1
            ),
        }
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = FirstFitAllocator(problem, ordering=FirstFitIcOrdering.PRICE_ASCENDING)
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        assert sol.cost == Currency("0.8/3600 usd")
        assert sum(sol.alloc.vms.values()) == 4
        assert sum(sol.alloc.containers.values()) == 4


class TestFirstFit3apps:
    """Tests for the first-fit allocator with 3 apps."""

    # 0.1 GB of memory for containers
    @pytest.mark.parametrize("system_3apps", [0.1], indirect=True)
    def test_first_fit_3apps_feasible(self, system_3apps) -> None:
        """Tests that the first-fit allocator works with 3 apps. Memory for
        containers is very small, so a feasible solution can be found."""
        system = system_3apps

        apps = system.apps
        workloads = {
            apps[0]: Workload(
                num_reqs=Requests("22.5 req"), time_slot_size=Time("s"), app=apps[0]
            ),
            apps[1]: Workload(
                num_reqs=Requests("15.3 req"), time_slot_size=Time("s"), app=apps[1]
            ),
            apps[2]: Workload(
                num_reqs=Requests("8 req"), time_slot_size=Time("s"), app=apps[2]
            ),
        }
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = FirstFitAllocator(problem, ordering=FirstFitIcOrdering.CORE_DESCENDING)

        sol = alloc.solve()

        assertions.assertEqual(sol.solving_stats.status, Status.INTEGER_FEASIBLE)
        assertions.assertEqual(sum(sol.alloc.vms.values()), 3)
        # Check that the number of cores in total is 14
        cores = 0
        for vm, used in sol.alloc.vms.items():
            if used:
                cores += vm.ic.cores
        assertions.assertEqual(cores, ComputationalUnits("14 cores"))
        assertions.assertEqual(sum(sol.alloc.containers.values()), 36)
        assertions.assertEqual(sol.cost, Currency("(0.8 + 0.4 + 0.2)/3600 usd"))

    # 2 GB of memory for containers
    @pytest.mark.parametrize("system_3apps", [2], indirect=True)
    def test_first_fit_3apps_infeasible(self, system_3apps) -> None:
        """Tests that the first-fit allocator works with 3 apps. Memory for
        containers is very big, so the first-fit allocator does not find a
        solution."""
        system = system_3apps

        apps = system.apps
        workloads = {
            apps[0]: Workload(
                num_reqs=Requests("22.5 req"), time_slot_size=Time("s"), app=apps[0]
            ),
            apps[1]: Workload(
                num_reqs=Requests("15.3 req"), time_slot_size=Time("s"), app=apps[1]
            ),
            apps[2]: Workload(
                num_reqs=Requests("8 req"), time_slot_size=Time("s"), app=apps[2]
            ),
        }
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = FirstFitAllocator(problem, ordering=FirstFitIcOrdering.CORE_DESCENDING)

        sol = alloc.solve()

        assertions.assertEqual(sol.solving_stats.status, Status.INFEASIBLE)
