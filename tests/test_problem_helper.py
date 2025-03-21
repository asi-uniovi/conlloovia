"""Tests for the `ProblemHelper` class."""

from cloudmodel.unified.units import (
    Requests,
    Time,
)

from conlloovia.model import (
    Workload,
    Problem,
)
from conlloovia import problem_helper


class TestSystem1ic1cc:
    """Basic tests with only one instance class and one container class."""

    def __set_up(self, system) -> Problem:
        """Sets up the problem."""
        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        return Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

    def test_create_vms_dict(self, system_1ic_1cc_1app) -> None:
        """Tests that the vms dictionary is created correctly."""
        system = system_1ic_1cc_1app
        problem = self.__set_up(system)

        vms_dict = problem_helper.create_vms_dict(problem.system.ics)

        assert list(vms_dict.keys()) == [
            "m5.xlarge-0",
            "m5.xlarge-1",
            "m5.xlarge-2",
            "m5.xlarge-3",
            "m5.xlarge-4",
        ]

        for vm in vms_dict.values():
            assert vm.ic == system.ics[0]

    def test_create_containers_dict(self, system_1ic_1cc_1app) -> None:
        """Tests that the containers dictionary is created correctly."""
        problem = self.__set_up(system_1ic_1cc_1app)

        vms_dict = problem_helper.create_vms_dict(problem.system.ics)
        containers_dict = problem_helper.create_containers_dict(
            problem.system.ccs, vms_dict
        )

        assert list(containers_dict.keys()) == [
            "m5.xlarge-0-1c2g",
            "m5.xlarge-0-0.5c0.5g-no-perf",
            "m5.xlarge-1-1c2g",
            "m5.xlarge-1-0.5c0.5g-no-perf",
            "m5.xlarge-2-1c2g",
            "m5.xlarge-2-0.5c0.5g-no-perf",
            "m5.xlarge-3-1c2g",
            "m5.xlarge-3-0.5c0.5g-no-perf",
            "m5.xlarge-4-1c2g",
            "m5.xlarge-4-0.5c0.5g-no-perf",
        ]

        for container in containers_dict.values():
            assert container.cc in problem.system.ccs


class TestSystem2ic2cc:
    """Basic tests with two instance classes and two container classes."""

    def __set_up(self, system) -> Problem:
        """Sets up the problem."""
        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        return Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

    def test_create_vms_dict(self, system_2ic_2cc_1app) -> None:
        """Tests that the vms dictionary is created correctly."""
        problem = self.__set_up(system_2ic_2cc_1app)

        vms_dict = problem_helper.create_vms_dict(problem.system.ics)

        assert list(vms_dict.keys()) == [
            "m5.large-0",
            "m5.large-1",
            "m5.large-2",
            "m5.large-3",
            "m5.large-4",
            "m5.xlarge-0",
            "m5.xlarge-1",
            "m5.xlarge-2",
            "m5.xlarge-3",
            "m5.xlarge-4",
        ]

        for vm in vms_dict.values():
            assert vm.ic in problem.system.ics

    def test_create_containers_dict(self, system_2ic_2cc_1app) -> None:
        """Tests that the containers dictionary is created correctly."""
        problem = self.__set_up(system_2ic_2cc_1app)

        vms_dict = problem_helper.create_vms_dict(problem.system.ics)
        containers_dict = problem_helper.create_containers_dict(
            problem.system.ccs, vms_dict
        )

        assert list(containers_dict.keys()) == [
            "m5.large-0-1c2g",
            "m5.large-0-2c2g",
            "m5.large-1-1c2g",
            "m5.large-1-2c2g",
            "m5.large-2-1c2g",
            "m5.large-2-2c2g",
            "m5.large-3-1c2g",
            "m5.large-3-2c2g",
            "m5.large-4-1c2g",
            "m5.large-4-2c2g",
            "m5.xlarge-0-1c2g",
            "m5.xlarge-0-2c2g",
            "m5.xlarge-1-1c2g",
            "m5.xlarge-1-2c2g",
            "m5.xlarge-2-1c2g",
            "m5.xlarge-2-2c2g",
            "m5.xlarge-3-1c2g",
            "m5.xlarge-3-2c2g",
            "m5.xlarge-4-1c2g",
            "m5.xlarge-4-2c2g",
        ]

    def test_get_vms_ordered_by_cores_desc(self, system_1ic_1cc_1app) -> None:
        """Tests that the vms are ordered correctly."""
        problem = self.__set_up(system_1ic_1cc_1app)

        vms_dict = problem_helper.create_vms_dict(problem.system.ics)
        vms_ordered = problem_helper.get_vms_ordered_by_cores_desc(vms_dict)

        for vm_i in range(len(vms_ordered) - 1):
            assert vms_ordered[vm_i].ic.cores <= vms_ordered[vm_i + 1].ic.cores

    def test_compute_cheapest_ic(self, system_2ic_2cc_1app) -> None:
        """Tests that the cheapest ic is computed correctly."""
        problem = self.__set_up(system_2ic_2cc_1app)

        cheapest_ic = problem_helper.compute_cheapest_ic(problem.system.ics)

        assert cheapest_ic == problem.system.ics[0]
