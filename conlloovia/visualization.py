"""This module provides ways of visualizing problems and solutions for
conlloovia."""

from rich.console import Console
from rich.table import Table, Column
from rich import print

from .model import Solution, Status, Problem


class SolutionPrettyPrinter:
    """Utilty methods to create pretty presentations of solutions."""

    def __init__(self, sol: Solution):
        self.sol = sol
        self.console = Console()

    def get_summary(self) -> str:
        """Returns a summary of the solution."""
        if self.is_infeasible_sol():
            return f"Non feasible solution. [bold red]{self.sol.solving_stats.status}"

        res = f"\nTotal cost: {self.sol.cost}"

        return res

    def print(self):
        """Prints a tables and a summary of the solution."""
        if self.is_infeasible_sol():
            print(f"Non feasible solution. [bold red]{self.sol.solving_stats.status}")
            return

        print(self.get_ic_table())
        print(self.get_cc_table())
        print(self.get_summary())

    def get_ic_table(self) -> Table:
        """Returns a Rich table with information about the instance classes."""
        if self.is_infeasible_sol():
            return Table(
                title=f"Non feasible solution. [bold red]{self.sol.solving_stats.status}"
            )

        table = Table(
            "VM",
            Column(header="Cost", justify="right"),
            title="VM allocation (only used VMs)",
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
        if self.is_infeasible_sol():
            return Table(
                title=f"Non feasible solution. [bold red]{self.sol.solving_stats.status}"
            )

        table = Table(
            "VM",
            "Container",
            "App",
            Column(header="Perf", justify="right"),
            title="Container allocation (only used VMs)",
        )

        alloc = self.sol.alloc

        total_num_containers = 0
        total_num_vms = 0
        prev_vm = None
        for container, num_replicas in alloc.containers.items():
            if num_replicas == 0:
                continue

            total_num_containers += num_replicas

            vm = container.vm
            cc = container.cc
            ic = vm.ic
            app = cc.app

            perf = self.sol.problem.system.perfs[ic, cc]
            perf = (perf * self.sol.problem.sched_time_size).to_reduced_units()

            if vm != prev_vm:
                total_num_vms += 1
                table.add_section()
                prev_vm = vm
                ic_col = f"{ic.name}[{vm.num}]"
            else:
                ic_col = ""

            table.add_row(
                ic_col, f"{cc.name} (x{int(num_replicas)})", app.name, str(perf)
            )

        table.add_section()
        table.add_row(
            f"total: {total_num_vms}",
            f"{int(total_num_containers)}",
            "",
        )

        return table

    def is_infeasible_sol(self):
        """Returns True if the solution is infeasible."""
        return self.sol.solving_stats.status not in [
            Status.OPTIMAL,
            Status.INTEGER_FEASIBLE,
        ]


class ProblemPrettyPrinter:
    """Utility functions to show pretty presentation of a problem."""

    def __init__(self, problem: Problem) -> None:
        self.problem: Problem = problem

    def print(self) -> None:
        """Prints information about the problem."""
        self.print_ics()
        self.print_ccs()
        self.print_apps()
        self.print_perfs()

    def table_ics(self) -> Table:
        """Returns a table with information about the instance classes."""
        table = Table(title="Instance classes")
        table.add_column("Instance class")
        table.add_column("Cores", justify="right")
        table.add_column("Mem", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Limit", justify="right")

        for ic in self.problem.system.ics:
            table.add_row(
                ic.name, str(ic.cores), str(ic.mem), str(ic.price), str(ic.limit)
            )

        return table

    def print_ics(self) -> None:
        """Prints information about the instance classes."""
        print(self.table_ics())

    def table_ccs(self) -> Table:
        """Returns a table with information about the container classes."""
        table = Table(title="Container classes")
        table.add_column("Container class")
        table.add_column("Cores", justify="right")
        table.add_column("Mem", justify="right")
        table.add_column("Limit", justify="right")

        for cc in self.problem.system.ccs:
            table.add_row(cc.name, str(cc.cores), str(cc.mem), str(cc.limit))

        return table

    def print_ccs(self) -> None:
        """Prints information about the container classes."""
        print(self.table_ccs())

    def table_apps(self) -> Table:
        """Returns a rich table with information about the apps, including the
        workload"""
        table = Table(title="Apps")
        table.add_column("Name")
        table.add_column("Workload", justify="right")

        for app in self.problem.system.apps:
            wl = self.problem.workloads[app]
            table.add_row(app.name, str(wl.num_reqs / wl.time_slot_size))

        return table

    def print_apps(self) -> None:
        """Prints information about the apps."""
        print(self.table_apps())

    def print_perfs(self) -> None:
        """Prints information about the performance."""
        table = Table(title="Performances")
        table.add_column("Instance class")
        table.add_column("Container class")
        table.add_column("App")
        table.add_column("RPS", justify="right")
        table.add_column("Price per million req.", justify="right")

        for ic in self.problem.system.ics:
            first = True
            for app in self.problem.system.apps:
                for cc in self.problem.system.ccs:
                    if app != cc.app or (ic, cc) not in self.problem.system.perfs:
                        continue  # Not all ICs handle all ccs

                    if first:
                        ic_column = f"{ic.name}"
                        first = False
                    else:
                        ic_column = ""

                    perf = self.problem.system.perfs[(ic, cc)]
                    price_per_1k_req = 1e6 * (ic.price.to("usd/h") / perf.to("req/h"))
                    table.add_row(
                        ic_column,
                        cc.name,
                        app.name,
                        str(perf.to("req/s").magnitude),
                        f"{price_per_1k_req.magnitude:.2f}",
                    )

            table.add_section()

        print(table)
