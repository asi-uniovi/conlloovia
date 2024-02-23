"""A simple example of how to use conlloovia."""

import logging

from cloudmodel.unified.units import (
    ComputationalUnits,
    CurrencyPerTime,
    Time,
    RequestsPerTime,
    Storage,
    Requests,
)

from conlloovia import (
    App,
    InstanceClass,
    ContainerClass,
    System,
    Workload,
    Problem,
    ConllooviaAllocator,
)
from conlloovia.visualization import SolutionPrettyPrinter

apps = (App(name="app0"),)

ics = (
    InstanceClass(
        name="m5.large",
        price=CurrencyPerTime("0.2 usd/hour"),
        cores=ComputationalUnits("1 core"),
        mem=Storage("8 gibibytes"),
        limit=5,  # Can be computed dynamically from the workload see below
    ),
    InstanceClass(
        name="m5.xlarge",
        price=CurrencyPerTime("0.4 usd/hour"),
        cores=ComputationalUnits("2 cores"),
        mem=Storage("16 gibibytes"),
        limit=5,  # Can be computed dynamically from the workload see below
    ),
)

ccs = (
    ContainerClass(
        name="1c2g",
        cores=ComputationalUnits("1 core"),
        mem=Storage("2 gibibytes"),
        app=apps[0],
    ),
    ContainerClass(
        name="2c2g",
        cores=ComputationalUnits("2 core"),
        mem=Storage("2 gibibytes"),
        app=apps[0],
    ),
)

base_perf = RequestsPerTime("1 req/s")
perfs = {
    (ics[0], ccs[0]): base_perf,
    (ics[0], ccs[1]): 1.5 * base_perf,
    (ics[1], ccs[0]): base_perf * 1.2,
    (ics[1], ccs[1]): 1.5 * base_perf * 1.2,
}

system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

app = system.apps[0]
workload = Workload(num_reqs=Requests("10 req"), time_slot_size=Time("1 s"), app=app)
workloads = {app: workload}

problem = Problem(system=system, workloads=workloads, sched_time_size=Time("1 s"))

logging.basicConfig(level=logging.INFO)

# If you want the limit to be computed dynamically from the workload, you can use
# the following code to create from the original problem a new problem with the limits:
#
# from conlloovia.limits import LimitsAdapter
#
# problem = LimitsAdapter(problem=problem).compute_adapted_problem()


alloc = ConllooviaAllocator(problem)
sol = alloc.solve()

SolutionPrettyPrinter(sol).print()
