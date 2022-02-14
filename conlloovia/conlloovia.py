import logging
from typing import List, Dict, Tuple

from pulp import LpVariable, lpSum, LpProblem, LpMinimize, LpStatus, value
from pulp.constants import LpBinary

from .model import Problem, Allocation, Vm, Container, Solution, Perf


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
        self.container_performances: Dict[str, Perf] = {}
        self.lPproblem = LpProblem("Container_problem", LpMinimize)

    def solve(self) -> Solution:
        """Solve the linear programming problem."""
        self.__create_vars()
        self.__create_objective()
        self.__create_restrictions()

        self.lPproblem.solve()

        return self.__create_solution()

    def __create_vars(self):
        """Creates the variables for the linear programming algorithm."""
        MAX_CONTAINERS = 10
        logging.warning(f"TODO: set a limit to max containers. Now: {MAX_CONTAINERS}")

        MAX_VCORES = 640
        logging.warning(f"TODO: MAX_VCORES: {MAX_VCORES}")

        for ic in self.problem.system.ics:
            limit = 5  # int(MAX_VCORES // ic.cores)
            logging.warning(f"TODO: limit for {ic.name}: {limit}")
            for i in range(limit):
                new_vm_name = f"{ic.name}-{i}"
                new_vm = Vm(ic=ic, num=i)
                self.vm_names.append(new_vm_name)
                self.vms[new_vm_name] = new_vm

                for j, cc in enumerate(self.problem.system.ccs):
                    for k in range(MAX_CONTAINERS):
                        new_container_name = f"{ic.name}-{i}-{cc.name}-{k}"
                        self.container_names.append(new_container_name)
                        self.containers[new_container_name] = Container(
                            cc=cc, vm=new_vm, num=k
                        )

                        perf = self.problem.system.perfs[(ic, cc)]
                        self.container_performances[new_container_name] = perf

        logging.info(
            f"There are {len(self.vm_names)} X variables and {len(self.container_names)} Z variables"
        )

        self.x = LpVariable.dicts(name="X", indexs=self.vm_names, cat=LpBinary)
        self.z = LpVariable.dicts(name="Z", indexs=self.container_names, cat=LpBinary)

    def __create_objective(self):
        """Adds the cost function to optimize."""
        self.lPproblem += lpSum(
            self.x[vm] * self.vms[vm].ic.price for vm in self.vm_names
        )

    def __create_restrictions(self):
        """Adds the performance restrictions."""
        for app in self.problem.system.apps:
            containers_for_this_app = []
            for name in self.container_names:
                if self.containers[name].cc.app == app:
                    containers_for_this_app.append(name)

            self.lPproblem += (
                lpSum(
                    self.z[name] * self.container_performances[name]
                    for name in containers_for_this_app
                )
                >= self.problem.workloads[app].value,  # TODO: units
                f"Enough_perf_for_{app}",
            )

        # Core, memory and IC  restrictions
        BIG_M = 1e6  # More than the maximum number of containers in an IC
        for vm_name in self.vm_names:
            containers_for_this_vm = []
            for container_name in self.container_names:
                if self.containers[container_name].vm == self.vms[vm_name]:
                    containers_for_this_vm.append(container_name)

            self.lPproblem += (
                lpSum(
                    self.z[container] * self.containers[container].cc.cores
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.cores,
                f"Enough_cores_in_vm_{vm_name}",
            )

            self.lPproblem += (
                lpSum(
                    self.z[container] * self.containers[container].cc.mem
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.mem,
                f"Enough_mem_in_vm_{vm_name}",
            )

            self.lPproblem += (
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
            problem=self.problem, alloc=alloc, cost=value(self.lPproblem.objective)
        )

        return sol

    def __log_solution(self):
        logging.info("Solution (only variables different to 0):")
        for i in self.x:
            if self.x[i].value() > 0:
                logging.info(f"  X_{i} = {self.x[i].value()}")
        logging.info("")

        for i in self.z:
            if self.z[i].value() > 0:
                logging.info(f"  Z_{i} = {self.z[i].value()}")

        logging.info("Status: %s", LpStatus[self.lPproblem.status])
        logging.info("Total cost: %f", value(self.lPproblem.objective))
