import logging

from conlloovia import (
    App,
    InstanceClass,
    ContainerClass,
    System,
    Workload,
    Problem,
    ConllooviaAllocator,
)

apps = [
    App(name="app0"),
]

ics = [
    InstanceClass(name="m5.large", price=0.2, cores=1, mem=8, limit=5),
    InstanceClass(name="m5.xlarge", price=0.4, cores=2, mem=16, limit=5),
]

ccs = [
    ContainerClass(name="1c2g", cores=1, mem=2, app=apps[0], limit=10),
    ContainerClass(name="2c2g", cores=2, mem=2, app=apps[0], limit=10),
]

base_perf = 1
perfs = {
    (ics[0], ccs[0]): base_perf,
    (ics[0], ccs[1]): 1.5 * base_perf,
    (ics[1], ccs[0]): base_perf * 1.2,
    (ics[1], ccs[1]): 1.5 * base_perf * 1.2,
}

system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

app = system.apps[0]
workload = Workload(value=10, app=app, time_unit="s")
workloads = {app: workload}
problem = Problem(system=system, workloads=workloads)

logging.basicConfig(level=logging.INFO)

alloc = ConllooviaAllocator(problem)
sol = alloc.solve()
