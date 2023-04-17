#!/usr/bin/env python
"""Tests for `ConllooviaAllocator`."""

import unittest

from click.testing import CliRunner
from pulp import PULP_CBC_CMD  # type: ignore

from cloudmodel.unified.units import (
    Currency,
    Time,
    Requests,
    RequestsPerTime,
)

from conlloovia.conlloovia import ConllooviaAllocator
from conlloovia.visualization import SolutionPrettyPrinter, ProblemPrettyPrinter
from conlloovia.model import (
    Workload,
    Problem,
    Status,
)
from conlloovia import cli

assertions = unittest.TestCase("__init__")


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "conlloovia.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


class TestSystem1ic1cc:
    """Basic tests with only one instance class and one container class"""

    def test_only_one(self, system_1ic_1cc_1app):
        """Tests that only one VM and container is required."""
        system = system_1ic_1cc_1app

        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        assert sol.solving_stats.status == Status.OPTIMAL
        assert sol.cost == Currency("0.2/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 1

    def test_only_one_with_solver_options(self, system_1ic_1cc_1app):
        """Tests that only one VM and container is required, using a solver with
        options."""
        system = system_1ic_1cc_1app

        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        alloc = ConllooviaAllocator(problem)
        solver = PULP_CBC_CMD(timeLimit=10, gapRel=0.01, threads=8)
        sol = alloc.solve(solver)

        assert sol.solving_stats.status == Status.OPTIMAL
        assert sol.cost == Currency("0.2/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 1

    def test_only_one_with_5s_window(self, system_1ic_1cc_1app):
        """Tests that only one VM and container is required, using 5 seconds
        window."""
        system = system_1ic_1cc_1app

        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("5s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(
            system=system, workloads=workloads, sched_time_size=Time("5s")
        )

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        print(alloc.lp_problem)
        assert sol.solving_stats.status == Status.OPTIMAL
        assert sol.cost == Currency("(0.2/3600)*5 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 1

    def test_1vm_2containers(self, system_1ic_1cc_1app):
        """Test that two containers on the same IC are used"""
        system = system_1ic_1cc_1app

        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("2 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        assert sol.cost == Currency("0.2/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 2

    def test_2vms_4containers(self, system_1ic_1cc_1app):
        """Test that two containers on the same IC are used"""
        system = system_1ic_1cc_1app

        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("4 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        assert sol.cost == Currency("0.4/3600 usd")
        assert sum(sol.alloc.vms.values()) == 2
        assert sum(sol.alloc.containers.values()) == 4


class TestSystem2ic2cc:
    """Basic tests with two instance classes and two container classes."""

    def __set_up(self, system, num_req) -> tuple[Workload, Problem]:
        """Sets up a problem with with the number of requests indicated."""
        app = system.apps[0]
        workload = Workload(
            num_reqs=Requests(f"{num_req} req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload}
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        ProblemPrettyPrinter(problem).print()
        return workload, problem

    def test_perf10(self, system_2ic_2cc_1app) -> None:
        """Tests that with 10 requests as workload."""
        system = system_2ic_2cc_1app
        workload, problem = self.__set_up(system, num_req=10)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        assert sol.cost == Currency("1.8/3600 usd")
        assert sum(sol.alloc.vms.values()) == 5

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assert len(vms_ics0) == 1

        vms_ics1 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[1] and sol.alloc.vms[vm])
        ]
        assert len(vms_ics1) == 4

        total_perf = sum(
            system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        assert (
            workload.num_reqs.magnitude
            <= total_perf.to(RequestsPerTime("req/s")).magnitude
        )

    def test_perf5(self, system_2ic_2cc_1app) -> None:
        """Tests that with 5 requests as workload."""
        system = system_2ic_2cc_1app
        workload, problem = self.__set_up(system, num_req=5)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        SolutionPrettyPrinter(sol).print()

        assertions.assertAlmostEqual(sol.cost, Currency("1/3600 usd"))
        assert sum(sol.alloc.vms.values()) == 3

        vms_ics0 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[0] and sol.alloc.vms[vm])
        ]
        assert len(vms_ics0) == 1

        vms_ics1 = [
            vm for vm in sol.alloc.vms if (vm.ic == system.ics[1] and sol.alloc.vms[vm])
        ]
        assert len(vms_ics1) == 2

        total_perf = sum(
            system.perfs[c.vm.ic, c.cc] * sol.alloc.containers[c]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        assert (
            workload.num_reqs.magnitude
            <= total_perf.to(RequestsPerTime("req/s")).magnitude
        )


class Test2apps:
    """Tests with two applications."""

    def test_2apps(self, system_2apps) -> None:
        """Tests that with 10 and 20 requests as workload."""
        system = system_2apps

        app0 = system.apps[0]
        app1 = system.apps[1]
        workloads = {
            app0: Workload(
                num_reqs=Requests("10 req"), time_slot_size=Time("s"), app=app0
            ),
            app1: Workload(
                num_reqs=Requests("20 req"), time_slot_size=Time("s"), app=app1
            ),
        }
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        assert sol.cost == Currency("0.8/3600 usd")
        assert sum(sol.alloc.vms.values()) == 1
        assert sum(sol.alloc.containers.values()) == 4


class TestInfeasible:
    """Basic infeasible test."""

    def test_infeasible(self, system_1ic_1cc_1app):
        """Tests that with 1000 requests as workload, which is infeasible."""
        system = system_1ic_1cc_1app

        app0 = system.apps[0]
        workloads = {
            app0: Workload(
                num_reqs=Requests("1000 req"), time_slot_size=Time("s"), app=app0
            ),
        }
        problem = Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        assertions.assertEqual(sol.solving_stats.status, Status.INFEASIBLE)
