#!/usr/bin/env python
import unittest

"""Tests for `conlloovia` package."""

import pytest

from click.testing import CliRunner

from conlloovia.conlloovia import ConllooviaAllocator
from conlloovia.model import (
    InstanceClass,
    App,
    ContainerClass,
    System,
    Workload,
    Problem,
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
            InstanceClass(name="m5.xlarge", price=0.2, cores=2, mem=16, limit=5),
        ]

        ccs = [
            ContainerClass(name="1c2g", cores=1, mem=2, app=apps[0], limit=10),
        ]

        base_perf = 1
        perfs = {
            (ics[0], ccs[0]): base_perf,
        }

        self.system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    def test_only_one(self):
        """Tests that only one VM and container is required"""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(value=1, app=app, time_unit="s")
        workloads = {app: workload_app}
        problem = Problem(system=self.system, workloads=workloads)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertEqual(sol.cost, 0.2)
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 1)

    def test_1vm_2containers(self):
        """Test that two containers on the same IC are used"""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(value=2, app=app, time_unit="s")
        workloads = {app: workload_app}
        problem = Problem(system=self.system, workloads=workloads)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertEqual(sol.cost, 0.2)
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 2)

    def test_2vms_4containers(self):
        """Test that two containers on the same IC are used"""
        self.__set_up()

        app = self.system.apps[0]
        workload_app = Workload(value=4, app=app, time_unit="s")
        workloads = {app: workload_app}
        problem = Problem(system=self.system, workloads=workloads)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertEqual(sol.cost, 0.4)
        self.assertEqual(sum(sol.alloc.vms.values()), 2)
        self.assertEqual(sum(sol.alloc.containers.values()), 4)


class TestSystem2ic2cc(unittest.TestCase):
    def __set_up(self):
        apps = [
            App(name="app0"),
        ]

        ics = [
            InstanceClass(name="m5.large", price=0.2, cores=1, mem=8, limit=5),
            InstanceClass(name="m5.xlarge", price=0.4, cores=2, mem=16, limit=5),
        ]

        ccs = [
            ContainerClass(name="1c2g", cores=1, mem=2, app=apps[0], limit=10),
            ContainerClass(name="2c2g", cores=2, mem=2, app=apps[0], limit=10),
        ]

        base_perf = 1
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
        workload = Workload(value=10, app=app, time_unit="s")
        workloads = {app: workload}
        problem = Problem(system=self.system, workloads=workloads)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertAlmostEqual(sol.cost, 1.8)
        self.assertEqual(sum(sol.alloc.vms.values()), 6)
        self.assertEqual(sum(sol.alloc.containers.values()), 9)

        vms_ics0 = [
            vm
            for vm in sol.alloc.vms
            if (vm.ic == self.system.ics[0] and sol.alloc.vms[vm])
        ]
        self.assertEqual(len(vms_ics0), 3)

        vms_ics1 = [
            vm
            for vm in sol.alloc.vms
            if (vm.ic == self.system.ics[1] and sol.alloc.vms[vm])
        ]
        self.assertEqual(len(vms_ics1), 3)

        total_perf = sum(
            self.system.perfs[c.vm.ic, c.cc]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        self.assertLessEqual(workload.value, total_perf)

    def test_perf5(self):
        self.__set_up()

        app = self.system.apps[0]
        workload = Workload(value=5, app=app, time_unit="s")
        workloads = {app: workload}
        problem = Problem(system=self.system, workloads=workloads)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertAlmostEqual(sol.cost, 1)
        self.assertEqual(sum(sol.alloc.vms.values()), 3)
        self.assertEqual(sum(sol.alloc.containers.values()), 4)

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
            self.system.perfs[c.vm.ic, c.cc]
            for c in sol.alloc.containers
            if sol.alloc.containers[c]
        )
        self.assertLessEqual(workload.value, total_perf)


class Test2apps(unittest.TestCase):
    def __set_up(self):
        apps = [
            App(name="app0"),
            App(name="app1"),
        ]

        ics = [
            InstanceClass(name="m5.large", price=0.2, cores=1, mem=8, limit=5),
            InstanceClass(name="m5.xlarge", price=0.4, cores=2, mem=16, limit=5),
            InstanceClass(name="m5.2xlarge", price=0.8, cores=4, mem=32, limit=5),
            InstanceClass(name="m5.4xlarge", price=1.6, cores=8, mem=64, limit=5),
        ]

        ccs = [
            ContainerClass(name="1c2gApp0", cores=1, mem=2, app=apps[0], limit=10),
            ContainerClass(name="2c2gApp0", cores=2, mem=2, app=apps[0], limit=10),
            ContainerClass(name="4c2gApp0", cores=4, mem=2, app=apps[0], limit=10),
            ContainerClass(name="1c2gApp1", cores=1, mem=2, app=apps[1], limit=10),
            ContainerClass(name="2c2gApp1", cores=2, mem=2, app=apps[1], limit=10),
            ContainerClass(name="1c4gApp1", cores=1, mem=4, app=apps[1], limit=10),
            ContainerClass(name="2c4gApp1", cores=2, mem=4, app=apps[1], limit=10),
            ContainerClass(name="1c8gApp1", cores=1, mem=8, app=apps[1], limit=10),
        ]

        base_perf_app0 = 5
        base_perf_app1 = 4
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
            app0: Workload(value=10, app=app0, time_unit="s"),
            app1: Workload(value=20, app=app0, time_unit="s"),
        }
        problem = Problem(system=self.system, workloads=workloads)

        alloc = ConllooviaAllocator(problem)
        sol = alloc.solve()

        self.assertAlmostEqual(sol.cost, 0.8)
        self.assertEqual(sum(sol.alloc.vms.values()), 1)
        self.assertEqual(sum(sol.alloc.containers.values()), 4)
