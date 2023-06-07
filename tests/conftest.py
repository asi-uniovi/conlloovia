"""Pytest configuration file for the conlloovia package."""

import pytest

from cloudmodel.unified.units import (
    ComputationalUnits,
    CurrencyPerTime,
    RequestsPerTime,
    Storage,
)

from conlloovia.model import (
    InstanceClass,
    App,
    ContainerClass,
    System,
)


@pytest.fixture(scope="module")
def system_1ic_1cc_1app() -> System:
    """Sets up a system with 1 instance class, 1 container class and 1 app."""
    apps = (App(name="app0"),)

    ics = (
        InstanceClass(
            name="m5.xlarge",
            price=CurrencyPerTime("0.2 usd/hour"),
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
        ),
    )

    base_perf = RequestsPerTime("1 req/s")
    perfs = {
        (ics[0], ccs[0]): base_perf,
    }

    system = System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)

    return system


@pytest.fixture(scope="module")
def system_2apps() -> System:
    """Sets up a system with 2 apps."""
    apps = (App(name="app0"), App(name="app1"))

    ics = (
        InstanceClass(
            name="m5.large",
            price=CurrencyPerTime("0.2 usd/hour"),
            cores=ComputationalUnits("1 cores"),
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
        InstanceClass(
            name="m5.2xlarge",
            price=CurrencyPerTime("0.8 usd/hour"),
            cores=ComputationalUnits("4 cores"),
            mem=Storage("32 gibibytes"),
            limit=5,
        ),
        InstanceClass(
            name="m5.4xlarge",
            price=CurrencyPerTime("1.6 usd/hour"),
            cores=ComputationalUnits("8 cores"),
            mem=Storage("64 gibibytes"),
            limit=5,
        ),
    )

    ccs = (
        ContainerClass(
            name="1c2gApp0",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[0],
        ),
        ContainerClass(
            name="2c2gApp0",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[0],
        ),
        ContainerClass(
            name="4c2gApp0",
            cores=ComputationalUnits("4 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[0],
        ),
        ContainerClass(
            name="1c2gApp1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[1],
        ),
        ContainerClass(
            name="2c2gApp1",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[1],
        ),
        ContainerClass(
            name="1c4gApp1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("4 gibibytes"),
            app=apps[1],
        ),
        ContainerClass(
            name="2c4gApp1",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("4 gibibytes"),
            app=apps[1],
        ),
        ContainerClass(
            name="1c8gApp1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("8 gibibytes"),
            app=apps[1],
        ),
    )

    base_perf_app0 = RequestsPerTime("5 req/s")
    base_perf_app1 = RequestsPerTime("4 req/s")
    perfs = {
        (ics[0], ccs[0]): base_perf_app0,
        (ics[0], ccs[1]): 1.2 * base_perf_app0,
        (ics[0], ccs[2]): 1.4 * base_perf_app0,
        (ics[1], ccs[0]): 1.5 * base_perf_app0,
        (ics[1], ccs[1]): 1.5 * 1.2 * base_perf_app0,
        (ics[1], ccs[2]): 1.5 * 1.4 * base_perf_app0,
        (ics[2], ccs[0]): 2 * base_perf_app0,
        (ics[2], ccs[1]): 2 * 1.2 * base_perf_app0,
        (ics[2], ccs[2]): 2 * 1.4 * base_perf_app0,
        (ics[3], ccs[0]): 4 * base_perf_app0,
        (ics[3], ccs[1]): 4 * 1.2 * base_perf_app0,
        (ics[3], ccs[2]): 4 * 1.4 * base_perf_app0,
        (ics[0], ccs[3]): base_perf_app1,
        (ics[0], ccs[4]): 1.2 * base_perf_app1,
        (ics[0], ccs[5]): 1.1 * base_perf_app1,
        (ics[0], ccs[6]): 1.3 * base_perf_app1,
        (ics[0], ccs[7]): 1.1 * base_perf_app1,
        (ics[1], ccs[3]): 1.5 * base_perf_app1,
        (ics[1], ccs[4]): 1.5 * 1.2 * base_perf_app1,
        (ics[1], ccs[5]): 1.5 * 1.1 * base_perf_app1,
        (ics[1], ccs[6]): 1.5 * 1.3 * base_perf_app1,
        (ics[1], ccs[7]): 1.5 * 1.1 * base_perf_app1,
        (ics[2], ccs[3]): 2 * base_perf_app1,
        (ics[2], ccs[4]): 2 * 1.2 * base_perf_app1,
        (ics[2], ccs[5]): 2 * 1.1 * base_perf_app1,
        (ics[2], ccs[6]): 2 * 1.3 * base_perf_app1,
        (ics[2], ccs[7]): 2 * 1.1 * base_perf_app1,
        (ics[3], ccs[3]): 4 * base_perf_app1,
        (ics[3], ccs[4]): 4 * 1.2 * base_perf_app1,
        (ics[3], ccs[5]): 4 * 1.1 * base_perf_app1,
        (ics[3], ccs[6]): 4 * 1.3 * base_perf_app1,
        (ics[3], ccs[7]): 4 * 1.1 * base_perf_app1,
    }

    return System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)


@pytest.fixture
def system_2ic_2cc_1app() -> System:
    """System with 2 instance classes and 2 container classes, 1 app."""
    apps = (App(name="app0"),)

    ics = (
        InstanceClass(
            name="m5.large",
            price=CurrencyPerTime("0.2 usd/hour"),
            cores=ComputationalUnits("1 cores"),
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
            cores=ComputationalUnits("1 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[0],
        ),
        ContainerClass(
            name="2c2g",
            cores=ComputationalUnits("2 cores"),
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

    return System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)


@pytest.fixture
def system_2ic_2cc_1app_small() -> System:
    """System with 2 instance classes and 2 container classes, 1 app."""
    apps = (App(name="app0"),)

    ics = (
        InstanceClass(
            name="m5.large",
            price=CurrencyPerTime("0.2 usd/hour"),
            cores=ComputationalUnits("1 cores"),
            mem=Storage("8 gibibytes"),
            limit=2,
        ),
        InstanceClass(
            name="m5.xlarge",
            price=CurrencyPerTime("0.4 usd/hour"),
            cores=ComputationalUnits("2 cores"),
            mem=Storage("16 gibibytes"),
            limit=2,
        ),
    )

    ccs = (
        ContainerClass(
            name="1c2g",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[0],
        ),
        ContainerClass(
            name="2c2g",
            cores=ComputationalUnits("2 cores"),
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

    return System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)


@pytest.fixture
def system_3apps(request):
    """Creates a system with 3 apps, 3 instance classes and 5 container
    classes. It has a paremeter request.mem_gb that must be fixed in the
    fixture."""
    mem_gb = request.param
    apps = (
        App(name="app0"),
        App(name="app1"),
        App(name="app2"),
    )

    # Notice that they are not sorted by cores. That's intentional, to test
    # that the allocator sorts them.
    ics = (
        InstanceClass(
            name="2c4g",
            price=CurrencyPerTime("0.2 usd/hour"),
            cores=ComputationalUnits("2 cores"),
            mem=Storage("4 gibibytes"),
            limit=3,
        ),
        InstanceClass(
            name="8c16g",
            price=CurrencyPerTime("0.8 usd/hour"),
            cores=ComputationalUnits("8 cores"),
            mem=Storage("16 gibibytes"),
            limit=1,
        ),
        InstanceClass(
            name="4c8g",
            price=CurrencyPerTime("0.4 usd/hour"),
            cores=ComputationalUnits("4 cores"),
            mem=Storage("8 gibibytes"),
            limit=2,
        ),
    )

    ccs = (
        # a0
        ContainerClass(
            name="150a0",
            cores=ComputationalUnits("0.150 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[0],
        ),
        ContainerClass(
            name="450a0",
            cores=ComputationalUnits("0.450 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[0],
        ),
        ContainerClass(
            name="300a0",
            cores=ComputationalUnits("0.300 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[0],
        ),
        # a1
        ContainerClass(
            name="650a1",
            cores=ComputationalUnits("0.650 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[1],
        ),
        ContainerClass(
            name="130a1",
            cores=ComputationalUnits("0.130 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[1],
        ),
        ContainerClass(
            name="260a1",
            cores=ComputationalUnits("0.260 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[1],
        ),
        # a2
        ContainerClass(
            name="1400a2",
            cores=ComputationalUnits("1.400 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[2],
        ),
        ContainerClass(
            name="2800a2",
            cores=ComputationalUnits("2.800 cores"),
            mem=Storage(f"{mem_gb} gibibytes"),
            app=apps[2],
        ),
    )

    reqs_per_sec_one_core = {
        apps[0]: RequestsPerTime("10.1 reqs/s"),
        apps[1]: RequestsPerTime("8.1 reqs/s"),
        apps[2]: RequestsPerTime("1 reqs/s"),
    }

    perfs = {}
    for ic in ics:
        for cc in ccs:
            cores = cc.cores.to("core").magnitude
            if ic.cores >= cc.cores and ic.mem >= cc.mem:
                perfs[(ic, cc)] = RequestsPerTime(
                    f"{reqs_per_sec_one_core[cc.app] * cores}"
                )

    return System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)
