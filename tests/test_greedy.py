"""Tests for `GreedyAllocator`."""

import unittest

from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    CurrencyPerTime,
    Time,
    Requests,
    RequestsPerTime,
    Storage,
)

from conlloovia.greedy import GreedyAllocator
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

    def test_only_one_greedy(self, system_1ic_1cc_1app) -> None:
        """Tests that only one VM and container is required, using the greedy
        allocator."""
        system = system_1ic_1cc_1app

        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        alloc = GreedyAllocator(problem)
        sol = alloc.solve()

        assert sol.solving_stats.status, Status.INTEGER_FEASIBLE
        assert sol.cost == Currency("0.2/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 1


class TestSystem2ic2cc:
    """Basic tests with two instance classes and two container classes."""

    def __solve_greedy(self, system: System, reqs: int) -> tuple[Workload, Solution]:
        """Solves the problem using the greedy allocator."""
        app = system.apps[0]
        workload = Workload(
            num_reqs=Requests(f"{reqs} req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = GreedyAllocator(problem)

        sol = alloc.solve()
        return workload, sol

    def test_perf1_greedy(self, system_2ic_2cc_1app) -> None:
        """Tests that only one VM from ics0 is required."""
        system = system_2ic_2cc_1app
        workload, sol = self.__solve_greedy(system, reqs=1)

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

    def test_perf4_greedy(self, system_2ic_2cc_1app) -> None:
        """Tests that 4 VMs of ics0 are required."""
        system = system_2ic_2cc_1app
        workload, sol = self.__solve_greedy(system, reqs=4)

        SolutionPrettyPrinter(sol).print()

        assertions.assertAlmostEqual(sol.cost, Currency("4*0.2/3600 usd"))
        assertions.assertEqual(sum(sol.alloc.vms.values()), 4)

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics0), 4)

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

    def test_perf5_greedy(self, system_2ic_2cc_1app) -> None:
        """Tests that 5 VMs of ics0 are required."""
        system = system_2ic_2cc_1app
        workload, sol = self.__solve_greedy(system, reqs=5)

        SolutionPrettyPrinter(sol).print()

        assertions.assertAlmostEqual(sol.cost, Currency("5*0.2/3600 usd"))
        assertions.assertEqual(sum(sol.alloc.vms.values()), 5)

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics0), 5)

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

    def test_perf6_greedy(self, system_2ic_2cc_1app):
        """This problem requires two VM types, since only the cheapest one
        would not be enough."""
        system = system_2ic_2cc_1app
        workload, sol = self.__solve_greedy(system, reqs=6)

        SolutionPrettyPrinter(sol).print()

        assertions.assertEqual(sol.solving_stats.status, Status.INTEGER_FEASIBLE)
        assertions.assertAlmostEqual(sol.cost, Currency("5*0.2/3600 usd + 1*0.4/3600 usd"))
        assertions.assertEqual(sum(sol.alloc.vms.values()), 6)

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics0), 5)

        vms_ics1 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[1] and sol.alloc.vms[vm])
        ]
        assertions.assertEqual(len(vms_ics1), 1)

        total_perf = sum(
            system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        assertions.assertLessEqual(
            workload.num_reqs.magnitude,
            total_perf.to(RequestsPerTime("req/s")).magnitude,
        )

    def test_perf7_greedy(self, system_2ic_2cc_1app_small):
        """This is not solvable, there are not enough resources."""
        system = system_2ic_2cc_1app_small
        workload, sol = self.__solve_greedy(system, reqs=8)

        SolutionPrettyPrinter(sol).print()

        assertions.assertEqual(sol.solving_stats.status, Status.INFEASIBLE)

class Test2apps:
    """Tests for the greedy allocator with 2 apps."""

    def test_2apps_greedy(self, system_2apps) -> None:
        """Test that the greedy allocator works with 2 apps."""
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

        alloc = GreedyAllocator(problem)
        sol = alloc.solve()

        assert sol.cost == Currency("0.8/3600 usd")
        assert sum(sol.alloc.vms.values()) == 4
        assert sum(sol.alloc.containers.values()) == 4


class TestGreedyMem(unittest.TestCase):
    """Test that the greedy allocator respects memory constraints."""

    def __set_up(self) -> System:
        """Creates a system with 2 instance classes and 1 container class."""
        apps = (App(name="app0"),)

        ics = (
            InstanceClass(
                name="2c8g",
                price=CurrencyPerTime("0.2 usd/hour"),
                cores=ComputationalUnits("2 cores"),
                mem=Storage("8 gibibytes"),
                limit=5,
            ),
            InstanceClass(
                name="4c16g",
                price=CurrencyPerTime("0.4 usd/hour"),
                cores=ComputationalUnits("4 cores"),
                mem=Storage("16 gibibytes"),
                limit=5,
            ),
        )

        ccs = (
            ContainerClass(
                name="1c8g",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("8 gibibytes"),
                app=apps[0],
                limit=10,
            ),
        )

        base_perf = RequestsPerTime("1 req/s")
        perfs = {
            (ics[0], ccs[0]): base_perf,
            (ics[1], ccs[0]): base_perf * 1.2,
        }

        return System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    def test_greedy_mem(self) -> None:
        """Two containers will be needed. Even though one 2c8g has enough cores
        for the two containers, two VMs are needed because of the memory."""
        system = self.__set_up()

        app = system.apps[0]
        workload = Workload(
            num_reqs=Requests("2 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = GreedyAllocator(problem)

        sol = alloc.solve()

        assertions.assertEqual(sum(sol.alloc.vms.values()), 2)
        assertions.assertEqual(sum(sol.alloc.containers.values()), 2)
        assertions.assertEqual(sol.cost, Currency("2*0.2/3600 usd"))
