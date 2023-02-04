"""A simple example of how to use conlloovia."""
import logging

from conlloovia import (
    App,
    InstanceClass,
    ContainerClass,
    System,
    Workload,
    Problem,
    ConllooviaAllocator,
    ureg,
)
from conlloovia.visualization import SolutionPrettyPrinter

Q_ = ureg.Quantity

apps = (App(name="app0"),)

ics = (
    InstanceClass(
        name="m5.large",
        price=Q_("0.2 usd/hour"),
        cores=Q_("1 core"),
        mem=Q_("8 gibibytes"),
        limit=5,
    ),
    InstanceClass(
        name="m5.xlarge",
        price=Q_("0.4 usd/hour"),
        cores=Q_("2 cores"),
        mem=Q_("16 gibibytes"),
        limit=5,
    ),
)

ccs = (
    ContainerClass(
        name="1c2g", cores=Q_("1 core"), mem=Q_("2 gibibytes"), app=apps[0], limit=10
    ),
    ContainerClass(
        name="2c2g", cores=Q_("2 core"), mem=Q_("2 gibibytes"), app=apps[0], limit=10
    ),
)

base_perf = Q_("1 req/s")
perfs = {
    (ics[0], ccs[0]): base_perf,
    (ics[0], ccs[1]): 1.5 * base_perf,
    (ics[1], ccs[0]): base_perf * 1.2,
    (ics[1], ccs[1]): 1.5 * base_perf * 1.2,
}

system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

app = system.apps[0]
workload = Workload(num_reqs=10, time_slot_size=Q_("s"), app=app)
workloads = {app: workload}
problem = Problem(system=system, workloads=workloads, sched_time_size=Q_("s"))

logging.basicConfig(level=logging.INFO)

alloc = ConllooviaAllocator(problem)
sol = alloc.solve()

SolutionPrettyPrinter(sol).print()
