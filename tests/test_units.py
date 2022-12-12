"""Tests about units for the `conlloovia` package."""
import unittest

import pytest

import pint

from conlloovia.model import (
    InstanceClass,
    Workload,
    ureg,
)

Q_ = ureg.Quantity


class TestUnits(unittest.TestCase):
    """Test that correct units work and that incorrect units give an error."""

    def test_valid_units_instance_class(self):
        """Checks a case with valid units for an intance class."""
        InstanceClass(
            name="ic0",
            price=Q_("1 usd/hour"),
            cores=Q_("1 core"),
            mem=Q_("1 megabytes"),
            limit=1,
        )

    def test_no_units_instance_class(self):
        """Checks a case with no units for the price of an intance class."""
        with pytest.raises(AttributeError):
            InstanceClass(
                name="ic0",
                price=1,
                cores=Q_("1 core"),
                mem=Q_("1 megabytes"),
                limit=1,
            )

    def test_invalid_price_units_instance_class(self):
        """Checks a case with invalid units for the price of an intance class."""
        with pytest.raises(pint.DimensionalityError):
            InstanceClass(
                name="ic0",
                price=Q_("1 meter"),
                cores=Q_("1 core"),
                mem=Q_("1 megabytes"),
                limit=1,
            )

    def test_invalid_cores_units_instance_class(self):
        """Checks a case with invalid units for the cores of an intance class."""
        with pytest.raises(pint.DimensionalityError):
            InstanceClass(
                name="ic0",
                price=Q_("1 usd/hour"),
                cores=Q_("1 meter"),
                mem=Q_("1 megabytes"),
                limit=1,
            )

    def test_invalid_mem_units_instance_class(self):
        """Checks a case with invalid units for the memory of an intance class."""
        with pytest.raises(pint.DimensionalityError):
            InstanceClass(
                name="ic0",
                price=Q_("1 usd/hour"),
                cores=Q_("1 core"),
                mem=Q_("1 meter"),
                limit=1,
            )

    def test_valid_workload_units_1s(self):
        """Checks a case with valid units for the workload."""
        Workload(num_reqs=1, time_slot_size=Q_("s"), app=None)

    def test_valid_workload_units_60s(self):
        """Checks a case with valid units for the workload."""
        Workload(num_reqs=1, time_slot_size=Q_("60s"), app=None)

    def test_invalid_workload_units(self):
        """Checks a case with invalid units for the workload."""
        with pytest.raises(pint.DimensionalityError):
            Workload(num_reqs=1, time_slot_size=Q_("meter"), app=None)
