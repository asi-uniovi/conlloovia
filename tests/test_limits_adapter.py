"""Tests for te LimitsAdapter class."""

from cloudmodel.unified.units import (
    Requests,
    Time,
)

from conlloovia.model import (
    Problem,
    Workload,
    System,
)

from conlloovia.conlloovia import LimitsAdapter


class TestsSystem1cc:
    """Basic tests with only one instance class and one container class. The performance
    of the container class is 1 rps, but there can be two containers in the instance
    class."""

    def __set_up(self, system) -> Problem:
        """Sets up the problem."""
        app = system.apps[0]
        workload_app = Workload(
            num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=app
        )
        workloads = {app: workload_app}
        return Problem(system=system, workloads=workloads, sched_time_size=Time("s"))

    def create_workloads(
        self, system: System, rps: Requests, time_slot_size: Time
    ) -> Workload:
        """Creates the workload dict required by the Problem class."""
        app = system.apps[0]
        workload_app = Workload(num_reqs=rps, time_slot_size=time_slot_size, app=app)
        return {app: workload_app}

    def test_adapt_limits_1(self, system_1ic_1cc_1app: System) -> None:
        """Tests that the limits are adapted correctly when they have to be 1 because the
        workload and the performance is 1 rps."""
        workloads = self.create_workloads(
            system=system_1ic_1cc_1app, rps=Requests("1 req"), time_slot_size=Time("s")
        )
        problem = Problem(
            system=system_1ic_1cc_1app, workloads=workloads, sched_time_size=Time("s")
        )

        adapted_problem = LimitsAdapter(problem=problem).compute_adapted_problem()

        assert adapted_problem.system.ics[0].limit == 1

    def test_adapt_limits_2(self, system_1ic_1cc_1app: System) -> None:
        """Tests that the limits are adapted correctly when they have to be 2 because
        the workload is 4 rps but the performance is 1 rps per cc and there can be 2
        containers in each VM."""
        workloads = self.create_workloads(
            system=system_1ic_1cc_1app, rps=Requests("4 req"), time_slot_size=Time("s")
        )
        problem = Problem(
            system=system_1ic_1cc_1app, workloads=workloads, sched_time_size=Time("s")
        )

        adapted_problem = LimitsAdapter(problem=problem).compute_adapted_problem()

        assert adapted_problem.system.ics[0].limit == 2

    def test_adapt_limits_3(self, system_1ic_1cc_1app: System) -> None:
        """Tests that the limits are adapted correctly when they have to be 3 because
        the workload is 5 rps but the performance is 1 rps per cc and there can be 2
        containers in each VM."""
        workloads = self.create_workloads(
            system=system_1ic_1cc_1app, rps=Requests("5 req"), time_slot_size=Time("s")
        )
        problem = Problem(
            system=system_1ic_1cc_1app, workloads=workloads, sched_time_size=Time("s")
        )

        adapted_problem = LimitsAdapter(problem=problem).compute_adapted_problem()

        assert adapted_problem.system.ics[0].limit == 3
