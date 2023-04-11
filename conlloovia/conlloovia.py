"""Main module of the conlloovia package. It defines the class
ConllooviaAllocator, which receives a conlloovia problem and constructs and
solves the corresponding linear programming problem using pulp. It also
defines GreedyAllocator, which is a simple greedy allocator that chooses the
cheapest instance class for each app in terms of performance per dollar.gr"""

import math
import logging
import os
import time
from typing import Any, Dict

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
    InstanceClass,
    ContainerClass,
    App,
    Allocation,
    Vm,
    Container,
    Solution,
    Status,
    SolvingStats,
)


def pulp_to_conlloovia_status(
    pulp_problem_status: int, pulp_solution_status: int
) -> Status:
    """Receives the PuLP status code for the problem (LpProblem.status) and the
    solution (LpProblem.sol_status) and returns a conlloovia Status."""
    if pulp_problem_status == pulp.LpStatusInfeasible:
        r = Status.INFEASIBLE
    elif pulp_problem_status == pulp.LpStatusNotSolved:
        r = Status.ABORTED
    elif pulp_problem_status == pulp.LpStatusOptimal:
        if pulp_solution_status == pulp.LpSolutionOptimal:
            r = Status.OPTIMAL
        else:
            r = Status.INTEGER_FEASIBLE
    elif pulp_problem_status == pulp.LpStatusUndefined:
        r = Status.INTEGER_INFEASIBLE
    else:
        r = Status.UNKNOWN
    return r


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
        self.container_performances: dict[str, float] = {}
        self.container_names_per_app: dict[App, list[Container]] = {}
        self.container_names_per_vm: dict[Vm, list[Container]] = {}

        for app in self.problem.system.apps:
            self.container_names_per_app[app] = []

        self.lp_problem = LpProblem("Container_problem", LpMinimize)
        self.x: LpVariable = LpVariable(name="x")  # Placeholders
        self.z: LpVariable = LpVariable(name="y")

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

        if status == Status.INTEGER_FEASIBLE:
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
            for i in range(ic.limit):
                new_vm_name = f"{ic.name}-{i}"
                new_vm = Vm(ic=ic, num=i)
                self.vm_names.append(new_vm_name)
                self.vms[new_vm_name] = new_vm
                self.container_names_per_vm[new_vm] = []

                for cc in self.problem.system.ccs:
                    perf = self.problem.system.perfs[(ic, cc)]
                    new_container_name = f"{ic.name}-{i}-{cc.name}"
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

        self.x = LpVariable.dicts(
            name="X", indices=self.vm_names, cat=LpBinary, lowBound=0
        )
        self.z = LpVariable.dicts(
            name="Z", indices=self.container_names, cat=LpInteger, lowBound=0
        )

    def __price_ic_window(self, ic: InstanceClass) -> float:
        """Returns the cost of the Instance Class in the scheduling window."""
        return (ic.price * self.problem.sched_time_size).to_reduced_units().magnitude

    def __create_objective(self) -> None:
        """Adds the cost function to optimize."""
        self.lp_problem += lpSum(
            self.x[vm] * self.__price_ic_window(self.vms[vm].ic) for vm in self.vm_names
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
                    self.z[name] * self.__perf_in_window(name)
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
                    self.z[container] * self.containers[container].cc.cores.magnitude
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.cores.magnitude * self.x[vm_name],
                f"Enough_cores_in_vm_{vm_name}",
            )

            # Memory restrictions
            self.lp_problem += (
                lpSum(
                    self.z[container] * self.containers[container].cc.mem.magnitude
                    for container in containers_for_this_vm
                )
                <= self.vms[vm_name].ic.mem.magnitude * self.x[vm_name],
                f"Enough_mem_in_vm_{vm_name}",
            )

    def __create_solution(self, solving_stats: SolvingStats) -> Solution:
        self.__log_solution(solving_stats)

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

        if solving_stats.status in [Status.OPTIMAL, Status.INTEGER_FEASIBLE]:
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
        for x in self.x.values():
            if x.value() > 0:
                logging.info("  %s = %i", x, x.value())
        logging.info("")

        for z in self.z.values():
            if z.value() > 0:
                logging.info("  %s = %i", z, z.value())

        logging.info("Total cost: %f", pulp.value(self.lp_problem.objective))
        logging.info("Solving stats: %s", solving_stats)


class GreedyAllocator:
    """Greedy allocator that selects the cheapest instance class (in terms of
    cores per dollar) and the smallest container class (in terms of cores)."""

    def __init__(self, problem: Problem) -> None:
        """Constructor.

        Args:
            problem: problem to solve"""
        self.problem = problem

        self.vms: dict[str, Vm] = {}
        self.containers: dict[str, Container] = {}

        start_creation = time.perf_counter()

        self.create_vms_and_containers()

        # Precalculate the cheapest instance class in terms of cores per dollar.
        # If there are several, select the one with the smallest number of
        # cores.
        self.cheapest_ic = self.compute_cheapest_ic()
        self.cheapest_vms = [
            vm for vm in self.vms.values() if vm.ic == self.cheapest_ic
        ]

        # Precompute the smallest container class for each app
        self.smallest_ccs_per_app: Dict[App, ContainerClass] = {}
        for app in problem.system.apps:
            ccs_for_app = [cc for cc in problem.system.ccs if cc.app == app]
            self.smallest_ccs_per_app[app] = min(ccs_for_app, key=lambda cc: cc.cores)

        end_creation = time.perf_counter()
        self.creation_time = end_creation - start_creation

    def compute_cheapest_ic(self):
        """Returns the cheapest instance class in terms of cores per dollar.
        If there are several, select the one with the smallest number of
        cores."""
        return min(
            self.problem.system.ics,
            key=lambda ic: (
                ic.price.to("usd/h") / ic.cores,
                ic.cores,
            ),
        )

    def create_vms_and_containers(self) -> None:
        for ic in self.problem.system.ics:
            for i in range(ic.limit):
                new_vm_name = f"{ic.name}-{i}"
                new_vm = Vm(ic=ic, num=i)
                self.vms[new_vm_name] = new_vm

                for cc in self.problem.system.ccs:
                    new_container_name = f"{ic.name}-{i}-{cc.name}"
                    new_container = Container(cc=cc, vm=new_vm)
                    self.containers[new_container_name] = new_container

    def compute_num_ccs_per_app(self) -> Dict[App, int]:
        """Computes the number of CCs needed for each app according to its
        workload."""
        num_ccs = {}
        for app in self.problem.system.apps:
            cc = self.smallest_ccs_per_app[app]
            cc_perf = self.problem.system.perfs[self.cheapest_ic, cc]
            cc_reqs_in_sched_ts = cc_perf * self.problem.sched_time_size

            num_ccs[app] = math.ceil(
                self.problem.workloads[app].num_reqs / cc_reqs_in_sched_ts
            )

        return num_ccs

    def create_infeasible_solution(self, start_solving: float) -> Solution:
        """Creates an infeasible solution. The parameter start_solving is the
        time when the solving started."""
        alloc = Allocation(
            vms=self.create_initial_vm_alloc(),
            containers=self.create_initial_container_alloc(),
        )

        sol = Solution(
            problem=self.problem,
            alloc=alloc,
            cost=Currency("0 usd"),
            solving_stats=SolvingStats(
                frac_gap=0,
                max_seconds=0,
                lower_bound=0,
                creation_time=self.creation_time,
                solving_time=time.perf_counter() - start_solving,
                status=Status.INFEASIBLE,
            ),
        )

        return sol

    def create_initial_vm_alloc(self) -> Dict[Vm, bool]:
        """Creates an initial VM allocation where no VM is allocated."""
        vm_alloc: Dict[Vm, bool] = {}
        for vm in self.vms.values():
            vm_alloc[vm] = False

        return vm_alloc

    def create_initial_container_alloc(self) -> Dict[Container, int]:
        """Creates an initial container allocation where no container is
        allocated."""
        container_alloc: Dict[Container, int] = {}
        for container in self.containers.values():
            container_alloc[container] = 0

        return container_alloc

    def solve(self) -> Solution:
        """Solves the problem using a Greedy algorithm.

        Returns:
            solution to the problem
        """

        start_solving = time.perf_counter()

        # Compute the number of CCs needed for each app according to its
        # workload
        num_ccs_per_app = self.compute_num_ccs_per_app()

        # Initialize the VM alloc to 0 for all VMs
        vm_alloc = self.create_initial_vm_alloc()

        # Initialize the container alloc to 0 for all containers
        container_alloc = self.create_initial_container_alloc()

        # Compute the number of VMs needed in total. Start assigning CCs and
        # when the number of cores of the VM exceeds the number of cores of the
        # cheapest instance class, create a new VM. Create the allocation and
        # compute the cost at the same time.
        num_vms = 0
        num_cores = 0  # Number of used cores of this VM
        current_vm = None
        cost = Currency("0 usd")
        for app in self.problem.system.apps:
            cc = self.smallest_ccs_per_app[app]
            logging.info(
                "Allocating %i %s CCs for app %s",
                num_ccs_per_app[app],
                cc.name,
                app.name,
            )

            if cc.cores > self.cheapest_ic.cores:
                logging.info(
                    "  Not enough cores for containers of app %s in the cheapest VM",
                    app.name,
                )
                return self.create_infeasible_solution(start_solving)

            for _ in range(num_ccs_per_app[app]):
                new_num_cores = num_cores + cc.cores

                if current_vm is None or new_num_cores > current_vm.ic.cores:
                    # We need a new VM
                    if num_vms == len(self.cheapest_vms):
                        logging.info("  Not enough VMs")
                        return self.create_infeasible_solution(start_solving)

                    current_vm = self.cheapest_vms[num_vms]
                    cost += current_vm.ic.price * self.problem.sched_time_size
                    vm_alloc[current_vm] = True

                    num_vms += 1
                    num_cores = 0

                    logging.info(
                        "  Using %i/%i VMs (%s)", num_vms, len(self.cheapest_vms), cost
                    )

                num_cores += cc.cores
                logging.info(
                    "    Using %s of VM %i (total of %s/%s)",
                    cc.cores,
                    num_vms - 1,
                    num_cores,
                    current_vm.ic.cores,
                )

                container = self.containers[
                    f"{current_vm.ic.name}-{current_vm.num}-{cc.name}"
                ]
                container_alloc[container] += 1
                if container_alloc[container] > cc.limit:
                    logging.info(
                        "  Not enough containers of class %s in VM %s-%i",
                        cc.name,
                        current_vm.ic.name,
                        current_vm.num,
                    )

                    # Another possibility would be to try to use the next CC
                    # and so on, but it's not worth it.
                    return self.create_infeasible_solution(start_solving)

        # Create the allocation
        alloc = Allocation(
            vms=vm_alloc,
            containers=container_alloc,
        )

        solving_stats = SolvingStats(
            frac_gap=0,
            max_seconds=0,
            lower_bound=0,
            creation_time=self.creation_time,
            solving_time=time.perf_counter() - start_solving,
            status=Status.INTEGER_FEASIBLE,
        )

        sol = Solution(
            problem=self.problem,
            alloc=alloc,
            cost=cost,
            solving_stats=solving_stats,
        )

        return sol


# pylint: disable = E, W, R, C
def _solve_CBC_patched(self, lp, use_mps=True):
    """Solve a MIP problem using CBC patched from original PuLP function
    to save a log with cbc's output and take from it the best bound."""

    def take_best_bound_from_log(filename, msg: bool):
        ret = None
        try:
            with open(filename, "r", encoding="utf8") as f:
                for l in f:
                    if msg:
                        print(l, end="")
                    if l.startswith("Lower bound:"):
                        ret = float(l.split(":")[-1])
        except:
            pass
        return ret

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
