"""This module contains functions to help creating heuristic algorithms from
conlloovia Problems."""

from typing import Dict

from .model import (
    App,
    Container,
    ContainerClass,
    InstanceClass,
    Vm,
)


def create_vms_dict(ics: tuple[InstanceClass, ...]) -> Dict[str, Vm]:
    """Creates a dictionary of VMs, indexed by their name."""
    vms = {}
    for ic in ics:
        for vm_num in range(ic.limit):
            new_vm = Vm(ic=ic, id_=vm_num)
            vms[new_vm.name()] = new_vm

    return vms


def create_containers_dict(
    ccs: tuple[ContainerClass, ...], vms: Dict[str, Vm]
) -> Dict[str, Container]:
    """Creates a dictionary of containers, indexed by their name."""
    containers = {}
    for vm in vms.values():
        for cc in ccs:
            new_container = Container(cc=cc, vm=vm)
            containers[new_container.name()] = new_container

    return containers


def get_vms_ordered_by_cores_desc(vms: Dict[str, Vm]) -> list[Vm]:
    """Returns a list of VMs ordered by decreasing number of cores. If they
    have the same number of cores, it orders them by cost."""
    return sorted(
        vms.values(),
        key=lambda vm: (-vm.ic.cores, vm.ic.price),
    )


def get_ccs_ordered_by_cores_and_mem_asc(
    ccs: tuple[ContainerClass, ...]
) -> list[ContainerClass]:
    """Returns a list of container classes ordered by increasing number of
    cores and memory."""
    return sorted(
        ccs,
        key=lambda cc: (cc.cores, cc.mem),
    )


def compute_cheapest_ic(ics: tuple[InstanceClass, ...]) -> InstanceClass:
    """Returns the cheapest instance class in terms of cores per dollar.
    If there are several, select the one with the smallest number of
    cores."""
    return min(
        ics,
        key=lambda ic: (
            ic.price.to("usd/h") / ic.cores,
            ic.cores,
        ),
    )


def create_empty_vm_alloc(vms: dict[str, Vm]) -> dict[Vm, bool]:
    """Creates a VM allocation where no VM is allocated."""
    vm_alloc: dict[Vm, bool] = {}
    for vm in vms.values():
        vm_alloc[vm] = False

    return vm_alloc


def create_empty_container_alloc(
    containers: dict[str, Container]
) -> dict[Container, int]:
    """Creates a container allocation where no container is
    allocated."""
    container_alloc: Dict[Container, int] = {}
    for container in containers.values():
        container_alloc[container] = 0

    return container_alloc


def get_ccs_for_app(
    ccs: tuple[ContainerClass, ...], app: App
) -> tuple[ContainerClass, ...]:
    """Returns a tuple of container classes for this app."""
    return tuple(cc for cc in ccs if cc.app == app)


def get_ccs_for_app_ordered_by_cores_and_mem_asc(
    ccs: tuple[ContainerClass, ...], app: App
) -> tuple[ContainerClass, ...]:
    """Returns a tuple of container classes for this app ordered by
    increasing number of cores and memory."""
    return tuple(
        sorted(
            get_ccs_for_app(ccs, app),
            key=lambda cc: (cc.cores, cc.mem),
        )
    )


def get_ics_ordered(ics: tuple[InstanceClass, ...]) -> list[InstanceClass]:
    """Sorts the instance classes according to their price per core, and
    in case of match, by the number of cores."""
    return sorted(
        ics,
        key=lambda ic: (
            ic.price.to("usd/h") / ic.cores,
            ic.cores,
        ),
    )


def get_apps_ordered_by_container_size_asc(
    ccs: tuple[ContainerClass, ...]
) -> list[App]:
    """Returns a list of apps ordered by increasing size of the smallest
    container for each app."""

    # Get all container classes ordered by ascending cores and memory
    ccs_ordered = get_ccs_ordered_by_cores_and_mem_asc(ccs)

    ordered_apps = []
    for cc in ccs_ordered:
        if cc.app not in ordered_apps:
            ordered_apps.append(cc.app)

    return ordered_apps
