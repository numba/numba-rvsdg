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
    scfg: "SCFG"
    sub_region_headers: dict[BlockName, list[RegionName]] = field(default_factory=dict, init=False)

    def __post_init__(self, name_gen):
        region_name = name_gen.new_region_name(kind=self.kind)
        object.__setattr__(self, "region_name", region_name)

    def __iter__(self):
        """Graph Iterator"""
        # initialise housekeeping datastructures
        to_visit, seen = [self.header], []
        while to_visit:
            # get the next block_name on the list
            block_name = to_visit.pop(0)
            # if we have visited this, we skip it
            if block_name in seen:
                continue
            else:
                seen.append(block_name)
            # yield the block_name
            yield block_name
            if block_name is self.exiting:
                continue
            # finally add any out_edges to the list of block_names to visit
            to_visit.extend(self.scfg.out_edges[block_name])

    @property
    def is_leaf_region(self):
        return len(self.subregions) == 0

    @property
    def is_root_region(self):
        return self in self.scfg.region_roots
