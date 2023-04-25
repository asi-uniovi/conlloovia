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
            limit=10,
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
            limit=10,
        ),
        ContainerClass(
            name="2c2gApp0",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[0],
            limit=10,
        ),
        ContainerClass(
            name="4c2gApp0",
            cores=ComputationalUnits("4 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[0],
            limit=10,
        ),
        ContainerClass(
            name="1c2gApp1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[1],
            limit=10,
        ),
        ContainerClass(
            name="2c2gApp1",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("2 gibibytes"),
            app=apps[1],
            limit=10,
        ),
        ContainerClass(
            name="1c4gApp1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("4 gibibytes"),
            app=apps[1],
            limit=10,
        ),
        ContainerClass(
            name="2c4gApp1",
            cores=ComputationalUnits("2 cores"),
            mem=Storage("4 gibibytes"),
            app=apps[1],
            limit=10,
        ),
        ContainerClass(
            name="1c8gApp1",
            cores=ComputationalUnits("1 cores"),
            mem=Storage("8 gibibytes"),
            app=apps[1],
            limit=10,
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
            limit=10,
        ),
        ContainerClass(
            name="2c2g",
            cores=ComputationalUnits("2 cores"),
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
            limit=10,
        ),
        ContainerClass(
            name="2c2g",
            cores=ComputationalUnits("2 cores"),
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

    return System(apps=apps, ics=ics, ccs=ccs, perfs=perfs)
