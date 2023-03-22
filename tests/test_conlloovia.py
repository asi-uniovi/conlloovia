#!/usr/bin/env python
"""Tests for `conlloovia` package."""

import unittest

from click.testing import CliRunner
from pulp import PULP_CBC_CMD  # type: ignore

from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    CurrencyPerTime,
    Time,
    Requests,
    RequestsPerTime,
    Storage,
)

from conlloovia.conlloovia import ConllooviaAllocator
from conlloovia.visualization import SolutionPrettyPrinter, ProblemPrettyPrinter
from conlloovia.model import (
    InstanceClass,
    App,
    ContainerClass,
    System,
    Workload,
    Problem,
    Status,
)
from conlloovia import cli


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "conlloovia.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


class TestSystem1ic1cc(unittest.TestCase):
    """Basic tests with only one instance class and one container class"""

    def __set_up(self):
        apps = [
            App(name="app0"),
        ]

        ics = [
            InstanceClass(
                name="m5.xlarge",
                price=CurrencyPerTime("0.2 usd/hour"),
                cores=ComputationalUnits("2 cores"),
                mem=Storage("16 gibibytes"),
                limit=5,
            ),
        ]

        ccs = [
            ContainerClass(
                name="1c2g",
                cores=ComputationalUnits("1 core"),
                mem=Storage("2 gibibytes"),
                app=apps[0],
                limit=10,
            ),
        ]

        base_perf = RequestsPerTime("1 req/s")
        perfs = {
            (ics[0], ccs[0]): base_perf,
        }

        self.system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    def test_only_one(self):
        """Tests that only one VM and container is required."""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        ProblemPrettyPrinter(problem).print()

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertEqual(sol.solving_stats.status, Status.OPTIMAL)
        self.assertEqual(sol.cost, Currency("0.2/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 1)

    def test_only_one_with_empty_solver(self):
        """Tests that only one VM and container is required, using a solver with
        no options."""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        alloc = ConllooviaAllocator(problem)
        solver = PULP_CBC_CMD()
        sol = alloc.solve(solver)

        self.assertEqual(sol.solving_stats.status, Status.OPTIMAL)
        self.assertEqual(sol.cost, Currency("0.2/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 1)

    def test_only_one_with_solver_options(self):
        """Tests that only one VM and container is required, using a solver with
        options."""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        alloc = ConllooviaAllocator(problem)
        solver = PULP_CBC_CMD(timeLimit=10, gapRel=0.01, threads=8)
        sol = alloc.solve(solver)

        self.assertEqual(sol.solving_stats.status, Status.OPTIMAL)
        self.assertEqual(sol.cost, Currency("0.2/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 1)

    def test_only_one_with_5s_window(self):
        """Tests that only one VM and container is required, using 5 seconds
        window."""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("5s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("5s")
        )

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        print(alloc.lp_problem)
        self.assertEqual(sol.solving_stats.status, Status.OPTIMAL)
        self.assertEqual(sol.cost, Currency("(0.2/3600)*5 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 1)

    def test_1vm_2containers(self):
        """Test that two containers on the same IC are used"""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("2 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertEqual(sol.cost, Currency("0.2/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 2)

    def test_2vms_4containers(self):
        """Test that two containers on the same IC are used"""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("4 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertEqual(sol.cost, Currency("0.4/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 2)
        self.assertEqual(sum(sol.alloc.containers.values()), 4)


class TestSystem2ic2cc(unittest.TestCase):
    def __set_up(self):
        apps = [
            App(name="app0"),
        ]

        ics = [
            InstanceClass(
                name="m5.large",
                price=CurrencyPerTime("0.2 usd/hour"),
                cores=ComputationalUnits("1 cores"),
                mem=Storage("8 gibibytes"),
                limit=5,
            ),
            InstanceClass(
                name="m5.xlarge",
                price=CurrencyPerTime("0.4 usd/hour"),
                cores=ComputationalUnits("2 cores"),
                mem=Storage("16 gibibytes"),
                limit=5,
            ),
        ]

        ccs = [
            ContainerClass(
                name="1c2g",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[0],
                limit=10,
            ),
            ContainerClass(
                name="2c2g",
                cores=ComputationalUnits("2 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[0],
                limit=10,
            ),
        ]

        base_perf = RequestsPerTime("1 req/s")
        perfs = {
            (ics[0], ccs[0]): base_perf,
            (ics[0], ccs[1]): 1.5 * base_perf,
            (ics[1], ccs[0]): base_perf * 1.2,
            (ics[1], ccs[1]): 1.5 * base_perf * 1.2,
        }

        self.system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    def test_perf10(self):
        self.__set_up()

        app = self.system.apps[0]
        workload = Workload(
            num_reqs=Requests("10 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        ProblemPrettyPrinter(problem).print()

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        self.assertAlmostEqual(sol.cost, Currency("1.8/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 5)

        vms_ics0 = [
            vm
            for vm in sol.alloc.vms
            if (vm.ic == self.system.ics[0] and sol.alloc.vms[vm])
        ]
        self.assertEqual(len(vms_ics0), 1)

        vms_ics1 = [
            vm
            for vm in sol.alloc.vms
            if (vm.ic == self.system.ics[1] and sol.alloc.vms[vm])
        ]
        self.assertEqual(len(vms_ics1), 4)

        total_perf = sum(
            self.system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        self.assertLessEqual(
            workload.num_reqs.magnitude,
            total_perf.to(RequestsPerTime("req/s")).magnitude,
        )

    def test_perf5(self):
        self.__set_up()

        app = self.system.apps[0]
        workload = Workload(
            num_reqs=Requests("5 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload}
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        self.assertAlmostEqual(sol.cost, Currency("1/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 3)

        vms_ics0 = [
            vm
            for vm in sol.alloc.vms
            if (vm.ic == self.system.ics[0] and sol.alloc.vms[vm])
        ]
        self.assertEqual(len(vms_ics0), 1)

        vms_ics1 = [
            vm
            for vm in sol.alloc.vms
            if (vm.ic == self.system.ics[1] and sol.alloc.vms[vm])
        ]
        self.assertEqual(len(vms_ics1), 2)

        total_perf = sum(
            self.system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        self.assertLessEqual(
            workload.num_reqs.magnitude,
            total_perf.to(RequestsPerTime("req/s")).magnitude,
        )


class Test2apps(unittest.TestCase):
    def __set_up(self):
        apps = [
            App(name="app0"),
            App(name="app1"),
        ]

        ics = [
            InstanceClass(
                name="m5.large",
                price=CurrencyPerTime("0.2 usd/hour"),
                cores=ComputationalUnits("1 cores"),
                mem=Storage("8 gibibytes"),
                limit=5,
            ),
            InstanceClass(
                name="m5.xlarge",
                price=CurrencyPerTime("0.4 usd/hour"),
                cores=ComputationalUnits("2 cores"),
                mem=Storage("16 gibibytes"),
                limit=5,
            ),
            InstanceClass(
                name="m5.2xlarge",
                price=CurrencyPerTime("0.8 usd/hour"),
                cores=ComputationalUnits("4 cores"),
                mem=Storage("32 gibibytes"),
                limit=5,
            ),
            InstanceClass(
                name="m5.4xlarge",
                price=CurrencyPerTime("1.6 usd/hour"),
                cores=ComputationalUnits("8 cores"),
                mem=Storage("64 gibibytes"),
                limit=5,
            ),
        ]

        ccs = [
            ContainerClass(
                name="1c2gApp0",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[0],
                limit=10,
            ),
            ContainerClass(
                name="2c2gApp0",
                cores=ComputationalUnits("2 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[0],
                limit=10,
            ),
            ContainerClass(
                name="4c2gApp0",
                cores=ComputationalUnits("4 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[0],
                limit=10,
            ),
            ContainerClass(
                name="1c2gApp1",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[1],
                limit=10,
            ),
            ContainerClass(
                name="2c2gApp1",
                cores=ComputationalUnits("2 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[1],
                limit=10,
            ),
            ContainerClass(
                name="1c4gApp1",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("4 gibibytes"),
                app=apps[1],
                limit=10,
            ),
            ContainerClass(
                name="2c4gApp1",
                cores=ComputationalUnits("2 cores"),
                mem=Storage("4 gibibytes"),
                app=apps[1],
                limit=10,
            ),
            ContainerClass(
                name="1c8gApp1",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("8 gibibytes"),
                app=apps[1],
                limit=10,
            ),
        ]

        base_perf_app0 = RequestsPerTime("5 req/s")
        base_perf_app1 = RequestsPerTime("4 req/s")
        perfs = {
            (ics[0], ccs[0]): base_perf_app0,
            (ics[0], ccs[1]): 1.2 * base_perf_app0,
            (ics[0], ccs[2]): 1.4 * base_perf_app0,
            (ics[1], ccs[0]): 1.5 * base_perf_app0,
            (ics[1], ccs[1]): 1.5 * 1.2 * base_perf_app0,
            (ics[1], ccs[2]): 1.5 * 1.4 * base_perf_app0,
            (ics[2], ccs[0]): 2 * base_perf_app0,
            (ics[2], ccs[1]): 2 * 1.2 * base_perf_app0,
            (ics[2], ccs[2]): 2 * 1.4 * base_perf_app0,
            (ics[3], ccs[0]): 4 * base_perf_app0,
            (ics[3], ccs[1]): 4 * 1.2 * base_perf_app0,
            (ics[3], ccs[2]): 4 * 1.4 * base_perf_app0,
            (ics[0], ccs[3]): base_perf_app1,
            (ics[0], ccs[4]): 1.2 * base_perf_app1,
            (ics[0], ccs[5]): 1.1 * base_perf_app1,
            (ics[0], ccs[6]): 1.3 * base_perf_app1,
            (ics[0], ccs[7]): 1.1 * base_perf_app1,
            (ics[1], ccs[3]): 1.5 * base_perf_app1,
            (ics[1], ccs[4]): 1.5 * 1.2 * base_perf_app1,
            (ics[1], ccs[5]): 1.5 * 1.1 * base_perf_app1,
            (ics[1], ccs[6]): 1.5 * 1.3 * base_perf_app1,
            (ics[1], ccs[7]): 1.5 * 1.1 * base_perf_app1,
            (ics[2], ccs[3]): 2 * base_perf_app1,
            (ics[2], ccs[4]): 2 * 1.2 * base_perf_app1,
            (ics[2], ccs[5]): 2 * 1.1 * base_perf_app1,
            (ics[2], ccs[6]): 2 * 1.3 * base_perf_app1,
            (ics[2], ccs[7]): 2 * 1.1 * base_perf_app1,
            (ics[3], ccs[3]): 4 * base_perf_app1,
            (ics[3], ccs[4]): 4 * 1.2 * base_perf_app1,
            (ics[3], ccs[5]): 4 * 1.1 * base_perf_app1,
            (ics[3], ccs[6]): 4 * 1.3 * base_perf_app1,
            (ics[3], ccs[7]): 4 * 1.1 * base_perf_app1,
        }

        self.system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    def test_2apps(self):
        self.__set_up()

        app0 = self.system.apps[0]
        app1 = self.system.apps[1]
        workloads = {
            app0: Workload(
                num_reqs=Requests("10 req"), time_slot_size=Time("s"), app=app0
            ),
            app1: Workload(
                num_reqs=Requests("20 req"), time_slot_size=Time("s"), app=app1
            ),
        }
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertAlmostEqual(sol.cost, Currency("0.8/3600 usd"))
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 4)


class TestInfeasible(unittest.TestCase):
    """Basic infeasible test."""

    def __set_up(self):
        apps = [
            App(name="app0"),
        ]

        ics = [
            InstanceClass(
                name="m5.xlarge",
                price=CurrencyPerTime("0.2 usd/hour"),
                cores=ComputationalUnits("2 cores"),
                mem=Storage("16 gibibytes"),
                limit=5,
            ),
        ]

        ccs = [
            ContainerClass(
                name="1c2g",
                cores=ComputationalUnits("1 cores"),
                mem=Storage("2 gibibytes"),
                app=apps[0],
                limit=10,
            ),
        ]

        base_perf = RequestsPerTime("1 req/s")
        perfs = {
            (ics[0], ccs[0]): base_perf,
        }

        self.system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    def test_infeasible(self):
        self.__set_up()

        app0 = self.system.apps[0]
        workloads = {
            app0: Workload(
                num_reqs=Requests("1000 req"), time_slot_size=Time("s"), app=app0
            ),
        }
        problem = Problem(
            system=self.system, workloads=workloads, sched_time_size=Time("s")
        )

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertEqual(sol.solving_stats.status, Status.INFEASIBLE)
