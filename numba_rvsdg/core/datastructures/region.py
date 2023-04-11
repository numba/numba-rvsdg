from dataclasses import dataclass, field, InitVar

from numba_rvsdg.core.datastructures.labels import NameGenerator


@dataclass(frozen=True)
class Region:
    name_gen: InitVar[NameGenerator]
    """Block Name Generator associated with this Block.
       Note: This is an initialization only argument and not
       a class attribute."""

    region_name: str = field(init=False)
    """Unique name identifier for this region"""

    # Regions can be meta, loop, head, tail, branch
    kind: str

    # The header block of this region
    header: str

    # The exiting block of this region
    exiting: str

    def __post_init__(self, name_gen: NameGenerator):
        region_name = name_gen.new_region_name(self.kind)
        object.__setattr__(self, "region_name", region_name)
