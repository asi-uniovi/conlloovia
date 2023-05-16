"""Tests for `FirstFitAllocator`."""

import unittest

import pytest
from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    CurrencyPerTime,
    Time,
    Requests,
    RequestsPerTime,
    Storage,
)

from conlloovia.first_fit import (
    FirstFitAllocator,
    FirstFitAllocator2,
    FirstFitIcOrdering,
)
from conlloovia.visualization import SolutionPrettyPrinter, ProblemPrettyPrinter
from conlloovia.model import (
    InstanceClass,
    App,
    ContainerClass,
    System,
    Workload,
    Problem,
    Solution,
    Status,
)

assertions = unittest.TestCase("__init__")


class TestSystem1ic1cc:
    """Basic tests with only one instance class and one container class."""

    def test_only_one_first_fit(self, system_1ic_1cc_1app):
        """Tests that only one VM and container is required, using the first-fit
        allocator."""
        system = system_1ic_1cc_1app

        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        alloc = FirstFitAllocator(problem)
        sol = alloc.solve()

        assert sol.solving_stats.status, Status.INTEGER_FEASIBLE
        assert sol.cost == Currency("0.2/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 1


class TestSystem2ic2cc:
    """Basic tests with two instance classes and two container classes."""

    def __solve_first_fit(self, system: System, reqs: int) -> tuple[Workload, Solution]:
        """Solves the problem using the first-fit allocator."""
        app = system.apps[0]
        workload = Workload(
            num_reqs=Requests(f"{reqs} req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = FirstFitAllocator(problem)

        sol = alloc.solve()
        return workload, sol

    def test_perf1_first_fit(self, system_2ic_2cc_1app):
        """Tests that only one VM from ics0 is required."""
        system = system_2ic_2cc_1app
        workload, sol = self.__solve_first_fit(system, reqs=1)

        SolutionPrettyPrinter(sol).print()

        assertions.assertAlmostEqual(sol.cost, Currency("0.2/3600 usd"))
        assertions.assertEqual(sum(sol.alloc.vms.values()), 1)

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics0), 1)

        vms_ics1 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[1] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics1), 0)

        total_perf = sum(
            system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        assertions.assertLessEqual(
            workload.num_reqs.magnitude,
            total_perf.to(RequestsPerTime("req/s")).magnitude,
        )

    def test_perf4_first_fit(self, system_2ic_2cc_1app):
        """Tests that 4 VMs of ics0 are required."""
        system = system_2ic_2cc_1app
        workload, sol = self.__solve_first_fit(system, reqs=4)

        SolutionPrettyPrinter(sol).print()

        assertions.assertAlmostEqual(sol.cost, Currency("2*0.4/3600 usd"))
        assertions.assertEqual(sum(sol.alloc.vms.values()), 2)

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics0), 0)

        vms_ics1 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[1] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics1), 2)

        total_perf = sum(
            system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        assertions.assertLessEqual(
            workload.num_reqs.magnitude,
            total_perf.to(RequestsPerTime("req/s")).magnitude,
        )

    def test_perf5_first_fit(self, system_2ic_2cc_1app):
        """Tests that 5 VMs of ics0 are required."""
        system = system_2ic_2cc_1app
        workload, sol = self.__solve_first_fit(system, reqs=5)

        SolutionPrettyPrinter(sol).print()

        assertions.assertAlmostEqual(
            sol.cost, Currency("(2*0.4/3600 usd) + (0.2/3600 usd)")
        )
        assertions.assertEqual(sum(sol.alloc.vms.values()), 3)

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics0), 1)

        vms_ics1 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[1] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics1), 2)

        total_perf = sum(
            system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        assertions.assertLessEqual(
            workload.num_reqs.magnitude,
            total_perf.to(RequestsPerTime("req/s")).magnitude,
        )

    def test_perf18_first_fit(self, system_2ic_2cc_1app):
        """Test that the problem is infeasible."""
        system = system_2ic_2cc_1app
        _, sol = self.__solve_first_fit(system, reqs=18)

        SolutionPrettyPrinter(sol).print()

        assertions.assertEqual(sol.solving_stats.status, Status.INFEASIBLE)


class Test2apps:
    """Tests for the first-fit allocator with 2 apps."""

    def test_2apps_first_fit(self, system_2apps):
        """Test that the first-fit allocator works with 2 apps."""
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

        alloc = FirstFitAllocator(problem)
        sol = alloc.solve()

        alloc.print_replica_info(compact_mode=True)

        assert sol.cost == Currency("1.6/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 2

    def test_2apps_first_fit2_cores_descending(self, system_2apps):
        """Test that the first-fit allocator v2 works with 2 apps using core
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

        alloc = FirstFitAllocator2(problem, ordering=FirstFitIcOrdering.CORE_DESCENDING)
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        assert sol.cost == Currency("1.6/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 2

    def test_2apps_first_fit2_price_ascending(self, system_2apps):
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

        alloc = FirstFitAllocator2(
            problem, ordering=FirstFitIcOrdering.PRICE_ASCENDING
        )
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        assert sol.cost == Currency("0.8/3600 usd")
        assert sum(sol.alloc.vms.values()) == 4
        assert sum(sol.alloc.containers.values()) == 4


class TestFirstFitMem(unittest.TestCase):
    """Test that the first-fit allocator respects memory constraints."""

    def __set_up(self):
        """Creates a system with 2 instance classes and 1 container class."""
        apps = [
            App(name="app0"),
        ]

        ics = [
            InstanceClass(
                name="2c4g",
                price=CurrencyPerTime("0.2 usd/hour"),
                cores=ComputationalUnits("2 cores"),
                mem=Storage("4 gibibytes"),
                limit=5,
            ),
            InstanceClass(
                name="4c8g",
                price=CurrencyPerTime("0.4 usd/hour"),
                cores=ComputationalUnits("4 cores"),
                mem=Storage("8 gibibytes"),
                limit=5,
            ),
        ]

        ccs = [
            ContainerClass(
                name="1c8g",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("8 gibibytes"),
                app=apps[0],
                limit=10,
            ),
        ]

        base_perf = RequestsPerTime("1 req/s")
        perfs = {
            (ics[0], ccs[0]): base_perf,
            (ics[1], ccs[0]): base_perf * 1.2,
        }

        return System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    def test_first_fit_mem(self):
        """Two containers will be needed. Even though one 5c8g has enough cores
        for the two containers, two VMs are needed because of the memory."""
        system = self.__set_up()

        app = system.apps[0]
        workload = Workload(
            num_reqs=Requests("2 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = FirstFitAllocator(problem)

        sol = alloc.solve()

        assertions.assertEqual(sum(sol.alloc.vms.values()), 2)
        assertions.assertEqual(sum(sol.alloc.containers.values()), 2)
        assertions.assertEqual(sol.cost, Currency("2*0.4/3600 usd"))


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

        alloc = FirstFitAllocator2(problem, ordering=FirstFitIcOrdering.CORE_DESCENDING)

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

        alloc = FirstFitAllocator2(problem, ordering=FirstFitIcOrdering.CORE_DESCENDING)

        sol = alloc.solve()

        assertions.assertEqual(sol.solving_stats.status, Status.INFEASIBLE)
