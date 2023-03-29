from dataclasses import dataclass, field, InitVar

from numba_rvsdg.core.datastructures.labels import NameGenerator, RegionName, BlockName


@dataclass(frozen=True)
class Region:
    name_gen: InitVar[NameGenerator]
    """Region Name Generator associated with this Region.
       Note: This is an initialization only argument and not
       a class attribute."""

    region_name: RegionName = field(init=False)
    """Unique name identifier for this region"""

    kind: str
    header: BlockName
    exiting: BlockName

    def __post_init__(self, name_gen):
        region_name = name_gen.new_region_name(kind=self.kind)
        object.__setattr__(self, "region_name", region_name)
