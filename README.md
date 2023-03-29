Conlloovia
==========

![Testing status](https://github.com/asi-uniovi/conlloovia/actions/workflows/tests.yaml/badge.svg)

Use linear programming to allocate containers to cloud infrastructure

Introduction
------------

Conlloovia is a Python package to solve allocation problems in Container as a
Service (CaaS) deployed on Infrastructure as a Service (IaaS) clouds. The
objective is to optimize the cost from the point of view of the cloud customer.

Inputs:

- A set of applications.
- A set of instance classes (VM types).
- A set of container classes (containers running applications).
- The performance of each container class in each instance class.
- The workload of each application.

All the inputs are represented by objects of the classes in the `conlloovia`
module and are collected in a `Problem` object.

Outputs:

- The allocation in a `Solution` object. An allocation indicates which
  containers are allocated to which instances, and how many instances of each
  type are used.

Installation
------------

Clone the repository:

```bash
git clone https://github.com/asi-uniovi/conlloovia.git
cd conlloovia
```

Optionally, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package with `pip`:

```bash
pip install .
```

If you want to be able to modify the code and see the changes reflected in the
package, install it in editable mode:

```bash
pip install -e .
```

Usage
-----

You can see [an example](examples/example1.py) of usage in the
[`examples`](examples) folder. You can run it with:

```bash
python examples/example.py
```

You will see the output of the solver at the end.

If you want to use the package in your own code, import the required classes:

```python
import logging  # Used to show the output of the solver at the end

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
    ureg,
)
from conlloovia.visualization import SolutionPrettyPrinter, ProblemPrettyPrinter
```

Create the objects that represent the system, the workload and the problem:

```python
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
```

You can print the problem with:

```python
ProblemPrettyPrinter(problem).print()
```

Optionally, set the logging level to `INFO` to see the output of the solver:

```python
logging.basicConfig(level=logging.INFO)
```

Finally, create an allocator and solve the problem:

```python
alloc = ConllooviaAllocator(problem)
sol = alloc.solve()
```

You can access the solution with the `sol` variable or you can print a summary
with:

```python
SolutionPrettyPrinter(sol).print()
```

Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)
project template.
