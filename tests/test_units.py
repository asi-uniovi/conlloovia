"""Tests about units for the `conlloovia` package."""
import unittest

import pytest

import pint

from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    CurrencyPerTime,
    Requests,
    Time,
    Storage,
)

from conlloovia.model import (
    InstanceClass,
    Workload,
)


class TestUnits(unittest.TestCase):
    """Test that correct units work and that incorrect units give an error."""

    def test_valid_units_instance_class(self):
        """Checks a case with valid units for an intance class."""
        InstanceClass(
            name="ic0",
            price=CurrencyPerTime("1 usd/hour"),
            cores=ComputationalUnits("1 core"),
            mem=Storage("1 megabytes"),
            limit=1,
        )

    def test_no_units_instance_class(self):
        """Checks a case with no units for the price of an intance class."""
        with pytest.raises(AttributeError):
            InstanceClass(
                name="ic0",
                price=1,
                cores=ComputationalUnits("1 core"),
                mem=Storage("1 megabytes"),
                limit=1,
            )

    def test_invalid_price_units_instance_class(self):
        """Checks a case with invalid units for the price of an intance class."""
        with pytest.raises(pint.DimensionalityError):
            InstanceClass(
                name="ic0",
                price=Time("1 s"),
                cores=ComputationalUnits("1 core"),
                mem=Storage("1 megabytes"),
                limit=1,
            )

    def test_invalid_cores_units_instance_class(self):
        """Checks a case with invalid units for the cores of an intance class."""
        with pytest.raises(pint.DimensionalityError):
            InstanceClass(
                name="ic0",
                price=CurrencyPerTime("1 usd/hour"),
                cores=Time("1 s"),
                mem=Storage("1 megabytes"),
                limit=1,
            )

    def test_invalid_mem_units_instance_class(self):
        """Checks a case with invalid units for the memory of an intance class."""
        with pytest.raises(pint.DimensionalityError):
            InstanceClass(
                name="ic0",
                price=CurrencyPerTime("1 usd/hour"),
                cores=ComputationalUnits("1 core"),
                mem=Time("1 s"),
                limit=1,
            )

    def test_valid_workload_units_1s(self):
        """Checks a case with valid units for the workload."""
        Workload(num_reqs=Requests("1 req"), time_slot_size=Time("s"), app=None)

    def test_valid_workload_units_60s(self):
        """Checks a case with valid units for the workload."""
        Workload(num_reqs=Requests("1 req"), time_slot_size=Time("60s"), app=None)

    def test_invalid_workload_units(self):
        """Checks a case with invalid units for the workload."""
        with pytest.raises(pint.DimensionalityError):
            Workload(
                num_reqs=Requests("1 req"), time_slot_size=Currency("1 usd"), app=None
            )
