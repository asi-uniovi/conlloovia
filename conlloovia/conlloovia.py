"""Main module of the conlloovia package. It defines the class
ConllooviaAllocator, which receives a conlloovia problem and constructs and
solves the corresponding linear programming problem using pulp."""

import logging
import math
import os
import time
from typing import Any

import pulp  # type: ignore

# type: ignore
from pulp import (
    LpVariable,
    lpSum,
    LpProblem,
    LpMinimize,
    PulpSolverError,
    COIN_CMD,
    log,
    subprocess,
    constants,
    warnings,
    operating_system,
    devnull,
)
from pulp.constants import LpBinary, LpInteger  # type: ignore

from cloudmodel.unified.units import Currency

from .model import (
    Problem,
    System,
    InstanceClass,
    App,
    Allocation,
    Vm,
    Container,
    ContainerClass,
    Solution,
    RequestsPerTime,
    Status,
    SolvingStats,
)


def pulp_to_conlloovia_status(
    pulp_problem_status: int, pulp_solution_status: int
) -> Status:
    """Receives the PuLP status code for the problem (LpProblem.status) and the
    solution (LpProblem.sol_status) and returns a conlloovia Status."""
    if pulp_problem_status == pulp.LpStatusInfeasible:
        res = Status.INFEASIBLE
    elif pulp_problem_status == pulp.LpStatusNotSolved:
        res = Status.ABORTED
    elif pulp_problem_status == pulp.LpStatusOptimal:
        if pulp_solution_status == pulp.LpSolutionOptimal:
            res = Status.OPTIMAL
        else:
            res = Status.INTEGER_FEASIBLE
    elif pulp_problem_status == pulp.LpStatusUndefined:
        res = Status.INTEGER_INFEASIBLE
    else:
        res = Status.UNKNOWN
    return res


class ConllooviaAllocator:
    """This class receives a problem of container allocation and gives methods
    to solve it and store the solution."""

    def __init__(self, problem: Problem) -> None:
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem

        self.vm_names: list[str] = []
        self.vms: dict[str, Vm] = {}

        self.container_names: list[str] = []
        self.containers: dict[str, Container] = {}
        self.container_performances: dict[str, RequestsPerTime] = {}
        self.container_names_per_app: dict[App, list[str]] = {}
        self.container_names_per_vm: dict[Vm, list[str]] = {}

        for app in self.problem.system.apps:
            self.container_names_per_app[app] = []

        self.lp_problem = LpProblem("Container_problem", LpMinimize)
        self.x_vars: LpVariable = LpVariable(name="x")  # Placeholders
        self.z_vars: LpVariable = LpVariable(name="y")

    def solve(self, solver: Any = None) -> Solution:
        """Solve the linear programming problem. A solver with options can be
        passed. For instance:

            from pulp import PULP_CBC_CMD
            solver = PULP_CBC_CMD(timeLimit=10, gapRel=0.01, threads=8)

        """
        start_creation = time.perf_counter()
        self.__create_vars()
        self.__create_objective()
        self.__create_restrictions()
        creation_time = time.perf_counter() - start_creation

        solving_stats = self.__solve_problem(solver, creation_time)
        return self.__create_solution(solving_stats)

    def __solve_problem(self, solver: Any, creation_time: float) -> SolvingStats:
        status = Status.UNKNOWN
        lower_bound = None

        start_solving = time.perf_counter()
        if solver is None:
            frac_gap = None
            max_seconds = None
        else:
            if "gapRel" in solver.optionsDict:
                frac_gap = solver.optionsDict["gapRel"]
            else:
                frac_gap = None
            max_seconds = solver.timeLimit

        try:
            self.lp_problem.solve(solver, use_mps=False)
        except PulpSolverError as exception:
            end_solving = time.perf_counter()
            solving_time = end_solving - start_solving
            status = Status.CBC_ERROR

            print(
                f"Exception PulpSolverError. Time to failure: {solving_time} seconds",
                exception,
            )
        else:
            # No exceptions
            end_solving = time.perf_counter()
            solving_time = time.perf_counter() - start_solving
            status = pulp_to_conlloovia_status(
                self.lp_problem.status, self.lp_problem.sol_status
            )

        if status in [Status.INTEGER_FEASIBLE, Status.ABORTED]:
            lower_bound = self.lp_problem.bestBound

        solving_stats = SolvingStats(
            frac_gap=frac_gap,
            max_seconds=max_seconds,
            lower_bound=lower_bound,
            creation_time=creation_time,
            solving_time=solving_time,
            status=status,
        )

        return solving_stats

    def __create_vars(self) -> None:
        """Creates the variables for the linear programming algorithm."""

        for ic in self.problem.system.ics:
            for vm_num in range(ic.limit):
                new_vm_name = f"{ic.name}-{vm_num}"
                new_vm = Vm(ic=ic, id_=vm_num)
                self.vm_names.append(new_vm_name)
                self.vms[new_vm_name] = new_vm
                self.container_names_per_vm[new_vm] = []

                for cc in self.problem.system.ccs:
                    if (ic, cc) not in self.problem.system.perfs:
                        continue  # This CC cannot be executed in this IC

                    perf = self.problem.system.perfs[(ic, cc)]
                    new_container_name = f"{ic.name}-{vm_num}-{cc.name}"
                    self.container_names.append(new_container_name)
                    new_container = Container(cc=cc, vm=new_vm)
                    self.containers[new_container_name] = new_container
                    self.container_names_per_app[cc.app].append(new_container_name)
                    self.container_names_per_vm[new_vm].append(new_container_name)
                    self.container_performances[new_container_name] = perf

        logging.info(
            "There are %d X variables and %d Z variables",
            len(self.vm_names),
            len(self.container_names),
        )

        self.x_vars = LpVariable.dicts(
            name="X", indices=self.vm_names, cat=LpBinary, lowBound=0
        )
        self.z_vars = LpVariable.dicts(
            name="Z", indices=self.container_names, cat=LpInteger, lowBound=0
        )

    def __price_ic_window(self, ic: InstanceClass) -> float:
        """Returns the cost of the Instance Class in the scheduling window."""
        return (ic.price * self.problem.sched_time_size).to_reduced_units().magnitude

    def __create_objective(self) -> None:
        """Adds the cost function to optimize."""
        self.lp_problem += lpSum(
            self.x_vars[vm] * self.__price_ic_window(self.vms[vm].ic)
            for vm in self.vm_names
        )

    def __perf_in_window(self, name: str) -> float:
        """Returns the number of requests that a container gives in the scheduling window.
        It receives the name of the variable."""
        perf_window = self.container_performances[name] * self.problem.sched_time_size
        return perf_window.to_reduced_units().magnitude

    def __create_restrictions(self) -> None:
        """Adds the problem restrictions."""

        # Enough performance
        for app in self.problem.system.apps:
            self.lp_problem += (
                lpSum(
                    self.z_vars[name] * self.__perf_in_window(name)
                    for name in self.container_names_per_app[app]
                )
                >= self.problem.workloads[app].num_reqs.magnitude,
                f"Enough_perf_for_{app.name}",
            )

        # Core and memory
        for vm_name in self.vm_names:
            containers_for_this_vm = self.container_names_per_vm[self.vms[vm_name]]

            # Core restrictions
            self.lp_problem += (
                lpSum(
                    self.z_vars[container]
                    * self.containers[container].cc.cores.magnitude
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.cores.magnitude * self.x_vars[vm_name],
                f"Enough_cores_in_vm_{vm_name}",
            )

            # Memory restrictions
            self.lp_problem += (
                lpSum(
                    self.z_vars[container] * self.containers[container].cc.mem.magnitude
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.mem.magnitude * self.x_vars[vm_name],
                f"Enough_mem_in_vm_{vm_name}",
            )

    def __create_solution(self, solving_stats: SolvingStats) -> Solution:
        self.__log_solution(solving_stats)

        vm_alloc = {}
        for vm_name in self.vm_names:
            vm = self.vms[vm_name]
            vm_alloc[vm] = self.x_vars[vm_name].value()

        container_alloc = {}
        for c_name in self.container_names:
            container = self.containers[c_name]
            container_alloc[container] = self.z_vars[c_name].value()

        alloc = Allocation(
            vms=vm_alloc,
            containers=container_alloc,
        )

        # The objective could be None if there is no workload
        if solving_stats.status in [Status.OPTIMAL, Status.INTEGER_FEASIBLE] and (
            pulp.value(self.lp_problem.objective) is not None
        ):
            cost = pulp.value(self.lp_problem.objective) * Currency("1 usd")
        else:
            cost = Currency("0 usd")

        sol = Solution(
            problem=self.problem,
            alloc=alloc,
            cost=cost,
            solving_stats=solving_stats,
        )

        return sol

    def __log_solution(self, solving_stats: SolvingStats) -> None:
        if solving_stats.status not in [Status.OPTIMAL, Status.INTEGER_FEASIBLE]:
            logging.info("No feasible solution. Solving stats: %s", solving_stats)
            return

        logging.info("Solution (only variables different to 0):")
        for var_x in self.x_vars.values():
            if var_x.value() > 0:
                logging.info("  %s = %i", var_x, var_x.value())
        logging.info("")

        for var_z in self.z_vars.values():
            if var_z.value() > 0:
                logging.info("  %s = %i", var_z, var_z.value())

        logging.info("Total cost: %f", pulp.value(self.lp_problem.objective))
        logging.info("Solving stats: %s", solving_stats)


class LimitsAdapter:
    """Class that adapts the limits of a problem to a new set of limits following this
    algorithm:

    1) For each container class, compute the maximum number of containers that can be
       allocated in each IC where it can run. The valid combinations of CCs and ICs can be
       obtained from the perfs dictionary.
    2) Compute the performance of each IC with that maximum number of containers for each
       container class.
    3) Obtain the performance of each IC for each app with the container class that
       maximizes the performance.
    4) Compute the number of VMs required to satisfy the workload with that performance
       for each app and each instance class.
    5) Compute the limits for each IC as the sum of the minimum number of VMs required for
       all the apps that use that IC.

    The new problem will have the same apps and container classes, but the instance
    classes will be the same as the original ones, but with the new limits.

    Usage: create an instance of this class with the problem to adapt and call
    compute_adapted_problem() to obtain the new problem.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def compute_adapted_problem(self) -> Problem:
        """Computes the adapted problem."""
        new_system = LimitsAdapter.create_system_with_new_limits(
            self.problem.system, self.compute_num_vms_per_ic_to_serve_all_wl()
        )
        new_problem = Problem(
            system=new_system,
            workloads=self.problem.workloads,
            sched_time_size=self.problem.sched_time_size,
        )
        return new_problem

    def compute_num_vms_per_ic_to_serve_all_wl(self) -> dict[InstanceClass, int]:
        """Computes the required VM to handle all the workload using only one instance
        class."""
        # 1) Compute the performance of each IC with the maximum number of containers
        perf_per_cc_maximized = {}
        for ic, cc in self.problem.system.perfs:
            max_containers = LimitsAdapter.compute_max_containers(ic, cc)

            # If the IC cannot run the CC, do not consider it
            if max_containers == 0:
                continue

            perf_per_cc_maximized[ic, cc] = (
                max_containers * self.problem.system.perfs[ic, cc]
            )

        # 2) and 3) Take for each app and instance class the container class with the
        # highest performance when maximized
        perf_per_app_ic_maximized = {}
        for ic, cc in perf_per_cc_maximized:
            if (ic, cc.app) not in perf_per_app_ic_maximized:
                # First time we see this IC for this app
                perf_per_app_ic_maximized[ic, cc.app] = perf_per_cc_maximized[ic, cc]
            else:
                if (
                    perf_per_cc_maximized[ic, cc]
                    > perf_per_app_ic_maximized[ic, cc.app]
                ):
                    # This CC has a higher performance than the previous maximum
                    perf_per_app_ic_maximized[ic, cc.app] = perf_per_cc_maximized[
                        ic, cc
                    ]

        # 4) Compute the number of VMs required for each app for each IC
        num_vms_per_app_per_ic = {}
        for (ic, app), perf in perf_per_app_ic_maximized.items():
            workload = (
                self.problem.workloads[app].num_reqs
                / self.problem.workloads[app].time_slot_size
            )
            if perf.to("rps").magnitude == 0:
                num_vms_per_app_per_ic[ic, app] = 0
            else:
                num_vms_per_app_per_ic[ic, app] = math.ceil(
                    workload.to("rps").magnitude / perf.to("rps").magnitude
                )

        # 5) Compute the required number of VMs for each IC as the sum of the number of
        # VMs required for each app that uses that IC
        num_vms_per_ic: dict[InstanceClass, int] = {}
        for ic in self.problem.system.ics:
            num_vms_per_ic[ic] = 0
            for app in self.problem.system.apps:
                if (ic, app) in num_vms_per_app_per_ic:
                    num_vms_per_ic[ic] += num_vms_per_app_per_ic[ic, app]

        return num_vms_per_ic

    @staticmethod
    def compute_max_containers(ic: InstanceClass, cc: ContainerClass):
        """Returns the maximum number of containers that can be allocated in the given IC
        for the given CC taking into account the cores and the memory."""
        max_containers_cores = int(ic.cores / cc.cores)

        # Some times, the memory is not considered and the container class has 0 memory
        if cc.mem.to("bytes").magnitude == 0:
            return max_containers_cores

        max_containers_mem = int(ic.mem / cc.mem)
        return min(max_containers_cores, max_containers_mem)

    @staticmethod
    def create_system_with_new_limits(
        system: System, new_limits: dict[InstanceClass, int]
    ):
        """Returns a new system with the limits replaced as specified in new_limits."""

        if not set(new_limits.keys()) == set(system.ics):
            raise ValueError(
                "The keys of new_limits must be all the instance classes in the original "
                "system"
            )

        # Create the new instance classes from the old ones, but with the new limits
        new_ic_tuple = tuple(
            InstanceClass(
                name=ic.name,
                price=ic.price,
                cores=ic.cores,
                mem=ic.mem,
                limit=new_limits[ic],
            )
            for ic in system.ics
            if new_limits[ic] > 0
        )

        # Create the new perfs using the new instance classes
        new_perfs = {}
        for ic, cc in system.perfs:
            for new_ic in new_ic_tuple:
                if new_ic.name == ic.name:
                    new_perfs[new_ic, cc] = system.perfs[ic, cc]
                    break

        new_system = System(
            apps=system.apps, ics=new_ic_tuple, ccs=system.ccs, perfs=new_perfs
        )
        return new_system


# pylint: disable = E, W, R, C
def _solve_CBC_patched(self, lp, use_mps=True):
    """Solve a MIP problem using CBC patched from original PuLP function
    to save a log with cbc's output and take from it the best bound."""

    def take_best_bound_from_log(filename, msg: bool):
        """Take the lower bound from the log file. If there is a line with "best possible"
        take the minimum between the lower bound and the best possible because the lower
        bound is only printed with three decimal digits."""
        lower_bound = None
        best_possible = None
        try:
            with open(filename, "r", encoding="utf8") as f:
                for l in f:
                    if msg:
                        print(l, end="")
                    if "best possible" in l:
                        # There are lines like this:
                        # Cbc0010I After 155300 nodes, 10526 on tree, 0.0015583333 best solution, best possible 0.0015392781 (59.96 seconds)
                        # or this: 'Cbc0005I Partial search - best objective 0.0015583333 (best possible 0.0015392781), took 5904080 iterations and 121519 nodes (60.69 seconds)\n'
                        try:
                            best_possible = float(
                                l.split("best possible")[-1]
                                .strip()
                                .split(" ")[0]
                                .split(")")[0]
                            )
                        except:
                            pass
                    if l.startswith("Lower bound:"):
                        lower_bound = float(l.split(":")[-1])
        except:
            pass
        if best_possible is not None and lower_bound is not None:
            return min(lower_bound, best_possible)
        return lower_bound

    if not self.executable(self.path):
        raise PulpSolverError(
            "Pulp: cannot execute %s cwd: %s" % (self.path, os.getcwd())
        )
    tmpLp, tmpMps, tmpSol, tmpMst = self.create_tmp_files(
        lp.name, "lp", "mps", "sol", "mst"
    )
    if use_mps:
        vs, variablesNames, constraintsNames, _ = lp.writeMPS(tmpMps, rename=1)
        cmds = " " + tmpMps + " "
        if lp.sense == constants.LpMaximize:
            cmds += "max "
    else:
        vs = lp.writeLP(tmpLp)
        # In the Lp we do not create new variable or constraint names:
        variablesNames = dict((v.name, v.name) for v in vs)
        constraintsNames = dict((c, c) for c in lp.constraints)
        cmds = " " + tmpLp + " "
    if self.optionsDict.get("warmStart", False):
        self.writesol(tmpMst, lp, vs, variablesNames, constraintsNames)
        cmds += "mips {} ".format(tmpMst)
    if self.timeLimit is not None:
        cmds += "sec %s " % self.timeLimit
    options = self.options + self.getOptions()
    for option in options:
        cmds += option + " "
    if self.mip:
        cmds += "branch "
    else:
        cmds += "initialSolve "
    cmds += "printingOptions all "
    cmds += "solution " + tmpSol + " "
    if self.msg:
        pipe = subprocess.PIPE  # Modified
    else:
        pipe = open(os.devnull, "w")
    logPath = self.optionsDict.get("logPath")
    if logPath:
        if self.msg:
            warnings.warn(
                "`logPath` argument replaces `msg=1`. The output will be redirected to the log file."
            )
        pipe = open(self.optionsDict["logPath"], "w")
    log.debug(self.path + cmds)
    args = []
    args.append(self.path)
    args.extend(cmds[1:].split())
    with open(tmpLp + ".log", "w", encoding="utf8") as pipe:
        print(f"You can check the CBC log at {tmpLp}.log", flush=True)
        if not self.msg and operating_system == "win":
            # Prevent flashing windows if used from a GUI application
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            cbc = subprocess.Popen(
                args, stdout=pipe, stderr=pipe, stdin=devnull, startupinfo=startupinfo
            )
        else:
            cbc = subprocess.Popen(args, stdout=pipe, stderr=pipe, stdin=devnull)

        # Modified to get the best bound
        # output, _ = cbc.communicate()
        # if pipe:
        #     print("CBC output")
        #     for line in StringIO(output.decode("utf8")):
        #         if line.startswith("Lower bound:"):
        #             lp.bestBound = float(line.split(":")[1].strip())

        #         print(line, end="")

        if cbc.wait() != 0:
            if pipe:
                pipe.close()
            raise PulpSolverError(
                "Pulp: Error while trying to execute, use msg=True for more details"
                + self.path
            )
        if pipe:
            pipe.close()
    if not os.path.exists(tmpSol):
        raise PulpSolverError("Pulp: Error while executing " + self.path)
    (
        status,
        values,
        reducedCosts,
        shadowPrices,
        slacks,
        sol_status,
    ) = self.readsol_MPS(tmpSol, lp, vs, variablesNames, constraintsNames)
    lp.assignVarsVals(values)
    lp.assignVarsDj(reducedCosts)
    lp.assignConsPi(shadowPrices)
    lp.assignConsSlack(slacks, activity=True)
    lp.assignStatus(status, sol_status)
    lp.bestBound = take_best_bound_from_log(tmpLp + ".log", self.msg)
    self.delete_tmp_files(tmpMps, tmpLp, tmpSol, tmpMst)
    return status


# Monkey patching
COIN_CMD.solve_CBC = _solve_CBC_patched
