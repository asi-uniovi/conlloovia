"""Main module of the conlloovia package. It defines the class ConllooviaAllocator,
which receives a conlloovia problem and constructs and solves the corresponding
linear programming problem using pulp."""


import logging
from typing import List, Dict

import pulp
from pulp import LpVariable, lpSum, LpProblem, LpMinimize, LpStatus, value
from pulp.constants import LpBinary

from .model import Problem, Allocation, Vm, Container, Solution, Status, ureg


def pulp_to_conlloovia_status(pulp_status: int) -> Status:
    """Receives a PuLP status code and returns a conlloovia Status."""
    if pulp_status == pulp.LpStatusInfeasible:
        r = Status.INFEASIBLE
    elif pulp_status == pulp.LpStatusNotSolved:
        r = Status.ABORTED
    elif pulp_status == pulp.LpStatusOptimal:
        r = Status.OPTIMAL
    elif pulp_status == pulp.LpStatusUndefined:
        r = Status.INTEGER_INFEASIBLE
    else:
        r = Status.UNKNOWN
    return r


class ConllooviaAllocator:
    """This class receives a problem of container allocation and gives methods
    to solve it and store the solution."""

    def __init__(self, problem: Problem):
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem

        self.vm_names: List[str] = []
        self.vms: Dict[str, Vm] = {}

        self.container_names: List[str] = []
        self.containers: Dict[str, Container] = {}
        self.container_performances: Dict[str, float] = {}

        self.lp_problem = LpProblem("Container_problem", LpMinimize)
        self.x: LpVariable = LpVariable(name="x")  # Placeholders
        self.z: LpVariable = LpVariable(name="y")

    def solve(self) -> Solution:
        """Solve the linear programming problem."""
        self.__create_vars()
        self.__create_objective()
        self.__create_restrictions()

        self.lp_problem.solve()

        return self.__create_solution()

    def __create_vars(self):
        """Creates the variables for the linear programming algorithm."""
        for ic in self.problem.system.ics:
            for i in range(ic.limit):
                new_vm_name = f"{ic.name}-{i}"
                new_vm = Vm(ic=ic, num=i)
                self.vm_names.append(new_vm_name)
                self.vms[new_vm_name] = new_vm

                for cc in self.problem.system.ccs:
                    for k in range(cc.limit):
                        new_container_name = f"{ic.name}-{i}-{cc.name}-{k}"
                        self.container_names.append(new_container_name)
                        self.containers[new_container_name] = Container(
                            cc=cc, vm=new_vm, num=k
                        )

                        perf = self.problem.system.perfs[(ic, cc)]
                        self.container_performances[new_container_name] = perf

        logging.info(
            "There are %d X variables and %d Z variables",
            len(self.vm_names),
            len(self.container_names),
        )

        self.x = LpVariable.dicts(name="X", indices=self.vm_names, cat=LpBinary)
        self.z = LpVariable.dicts(name="Z", indices=self.container_names, cat=LpBinary)

    def __create_objective(self):
        """Adds the cost function to optimize."""
        self.lp_problem += lpSum(
            self.x[vm]
            * self.vms[vm]
            .ic.price.to(ureg.usd / self.problem.sched_time_size)
            .magnitude
            for vm in self.vm_names
        )

    def __create_restrictions(self):
        """Adds the performance restrictions."""
        for app in self.problem.system.apps:
            containers_for_this_app = []
            for name in self.container_names:
                if self.containers[name].cc.app == app:
                    containers_for_this_app.append(name)

            self.lp_problem += (
                lpSum(
                    self.z[name]
                    * self.container_performances[name]
                    .to(ureg.req / self.problem.sched_time_size)
                    .magnitude
                    for name in containers_for_this_app
                )
                >= self.problem.workloads[app].num_reqs,
                f"Enough_perf_for_{app}",
            )

        # Core, memory and IC  restrictions
        BIG_M = 1e6  # More than the maximum number of containers in an IC
        for vm_name in self.vm_names:
            containers_for_this_vm = []
            for container_name in self.container_names:
                if self.containers[container_name].vm == self.vms[vm_name]:
                    containers_for_this_vm.append(container_name)

            self.lp_problem += (
                lpSum(
                    self.z[container] * self.containers[container].cc.cores.magnitude
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.cores.magnitude,
                f"Enough_cores_in_vm_{vm_name}",
            )

            self.lp_problem += (
                lpSum(
                    self.z[container] * self.containers[container].cc.mem.magnitude
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.mem.magnitude,
                f"Enough_mem_in_vm_{vm_name}",
            )

            self.lp_problem += (
                lpSum(self.z[container] for container in containers_for_this_vm)
                <= self.x[vm_name] * BIG_M,
                f"Enough_instances_in_vm_{vm_name}",
            )

    def __create_solution(self):
        self.__log_solution()

        vm_alloc = {}
        for vm_name in self.vm_names:
            vm = self.vms[vm_name]
            vm_alloc[vm] = self.x[vm_name].value()

        container_alloc = {}
        for c_name in self.container_names:
            container = self.containers[c_name]
            container_alloc[container] = self.z[c_name].value()

        alloc = Allocation(
            vms=vm_alloc,
            containers=container_alloc,
        )

        sol = Solution(
            problem=self.problem,
            alloc=alloc,
            cost=value(self.lp_problem.objective) * ureg.usd,
            status=pulp_to_conlloovia_status(self.lp_problem.status),
        )

        return sol

    def __log_solution(self):
        logging.info("Solution (only variables different to 0):")
        for x in self.x.values():
            if x.value() > 0:
                logging.info("  %s = %i", x, x.value())
        logging.info("")

        for z in self.z.values():
            if z.value() > 0:
                logging.info("  %s = %i", z, z.value())

        logging.info("Status: %s", LpStatus[self.lp_problem.status])
        logging.info("Total cost: %f", value(self.lp_problem.objective))
