"""This module provides ways of visualizing problems and solutions for
conlloovia."""

from rich.console import Console
from rich.table import Table
from rich import print
import pint

from .model import Solution, Status

ureg = pint.UnitRegistry()


class SolutionPrettyPrinter:
    """Utilty methods to create pretty presentations of solutions."""

    def __init__(self, sol: Solution):
        self.sol = sol
        self.console = Console()

    def get_summary(self) -> str:
        """Returns a summary of the solution."""
        if self.sol.solving_stats.status not in [
            Status.OPTIMAL,
            Status.INTEGER_FEASIBLE,
        ]:
            return f"Non feasible solution. [bold red]{self.sol.solving_stats.status}"

        res = f"\nTotal cost: {self.sol.cost}"

        return res

    def print(self):
        """Prints a table for each application and a summary of the solution."""
        print()

        print(
            f"Workloads (scheduling time window: {self.sol.problem.sched_time_size}):"
        )
        for app, workload in self.sol.problem.workloads.items():
            print(f"    {app.name}: {workload.num_reqs} requests")
        print()

        print(self.get_ic_table())
        print(self.get_cc_table())
        print(self.get_summary())

    def get_ic_table(self) -> Table:
        """Returns a Rich table with information about the instance classes."""
        table = Table(
            "VM",
            "Cost",
            title="Allocation (only used VMs)",
        )

        alloc = self.sol.alloc

        total_num_vms = 0
        total_cost = 0.0
        for vm, is_vm_allocated in alloc.vms.items():
            if not is_vm_allocated:
                continue

            total_num_vms += 1

            ic = vm.ic
            cost = (ic.price * self.sol.problem.sched_time_size).to_reduced_units()
            total_cost += cost

            table.add_row(f"{ic.name}[{vm.num}]", str(cost))

        table.add_section()

        table.add_row(
            f"total: {total_num_vms}",
            f"{total_cost}",
        )

        return table

    def get_cc_table(self) -> Table:
        """Returns a Rich table with information about the container classes."""
        table = Table(
            "VM",
            "Container",
            "App",
            "Perf",
            title="Allocation (only used containers)",
        )

        alloc = self.sol.alloc

        total_num_containers = 0
        total_num_vms = 0
        prev_vm = None
        for container, num_replicas in alloc.containers.items():
            if num_replicas == 0:
                continue

            total_num_containers += 1

            vm = container.vm
            cc = container.cc
            ic = vm.ic
            app = cc.app

            if vm != prev_vm:
                total_num_vms += 1
                table.add_section()
                prev_vm = vm
                table.add_row(f"{ic.name}[{vm.num}]", "", "")

            perf = self.sol.problem.system.perfs[ic, cc]
            perf = (perf * self.sol.problem.sched_time_size).to_reduced_units()
            table.add_row("", f"{cc.name}[{num_replicas}]", app.name, str(perf))

        table.add_section()
        table.add_row(
            f"total: {total_num_vms}",
            f"{total_num_containers}",
            "",
        )

        return table
