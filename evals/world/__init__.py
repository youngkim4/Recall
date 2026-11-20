"""The eval world: an invented iMessage history of about three years, with known facts.

See `people.py` for who is in it, the `scenes_*.py` files for the written
moments, `filler.py` for the everyday days, and `build.py` to turn it into
an archive.
"""

from evals.world.build import Built, World, WorldError, build_world, world, zone

__all__ = ["Built", "World", "WorldError", "build_world", "world", "zone"]
