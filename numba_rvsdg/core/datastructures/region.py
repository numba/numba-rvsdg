from dataclasses import dataclass, field
from numba_rvsdg.core.datastructures.basic_block import Block

from numba_rvsdg.core.datastructures.labels import NameGenerator, RegionName, BlockName


@dataclass(frozen=True)
class Region(Block):
    region_name: RegionName = field(init=False)
    """Unique name identifier for this region"""

    def __post_init__(self, name_gen: NameGenerator):
        region_name = name_gen.new_region_name(self.label)
        object.__setattr__(self, "region_name", region_name)


@dataclass(frozen=True)
class LoopRegion(Region):
    header: BlockName
    exiting: BlockName
    ...


@dataclass(frozen=True)
class MetaRegion(Region):
    ...


# TODO: Register new regions over here
region_types = {
    "loop": LoopRegion
}


def get_region_class(region_type_string: str):
    if region_type_string in region_types:
        return region_types[region_type_string]
    else:
        raise TypeError(f"Region Type {region_type_string} not recognized.")


def get_region_class_str(region: Region):
    for key, value in region_types.items():
        if isinstance(region, value):
            return key
    else:
        raise TypeError(f"Region Type of {region} not recognized.")
