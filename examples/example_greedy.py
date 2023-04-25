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
)
from conlloovia.visualization import SolutionPrettyPrinter
from conlloovia.greedy import GreedyAllocator

apps = (App(name="app0"),)

ics = (
    InstanceClass(
        name="m5.large",
        price=CurrencyPerTime("0.2 usd/hour"),
        cores=ComputationalUnits("1 core"),
        mem=Storage("8 gibibytes"),
        limit=5,
    ),
    InstanceClass(
        name="m5.xlarge",
        price=CurrencyPerTime("0.4 usd/hour"),
        cores=ComputationalUnits("2 cores"),
        mem=Storage("16 gibibytes"),
        limit=5,
    ),
)

ccs = (
    ContainerClass(
        name="1c2g",
        cores=ComputationalUnits("1 core"),
        mem=Storage("2 gibibytes"),
        app=apps[0],
        limit=10,
    ),
    ContainerClass(
        name="2c2g",
        cores=ComputationalUnits("2 core"),
        mem=Storage("2 gibibytes"),
        app=apps[0],
        limit=10,
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

alloc = GreedyAllocator(problem)
sol = alloc.solve()

SolutionPrettyPrinter(sol).print()
