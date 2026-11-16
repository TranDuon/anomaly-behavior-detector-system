"""Single registry of displayed / recorded behaviours.

The four core entries reproduce, value for value, what used to be written
inline in ``api/fight_server.py``, ``fight/mmaction_runtime.py`` and
``fight/display_geometry.py``; ``tests/test_behavior_registry.py`` pins them.
Extensions register extra entries after the core ones, so core ordering,
gates and recordings are never changed by adding a behaviour.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    display_label: str
    color_bgr: tuple = (50, 65, 235)
    confirm_seconds: float = 0.4
    release_seconds: float = 0.25
    alarm_enabled: bool = True
    kind: str = "core"  # "core" | "extension"
    # Core state is forced to "normal" when the runtime has no action region for it.
    reset_without_region: bool = True
    # An active / verifying state of this behaviour disables the small-person display path.
    blocks_small_person_display: bool = True
    description: str = ""

    def to_dict(self):
        value = asdict(self)
        value["color_bgr"] = list(self.color_bgr)
        return value


CORE_BEHAVIORS = (
    BehaviorSpec("strike", "STRIKE", confirm_seconds=0.4, description="AVA fight/hit a person"),
    BehaviorSpec("grapple", "GRAPPLE", confirm_seconds=0.4, alarm_enabled=False,
                 description="diagnostic only; pilot head not approved"),
    BehaviorSpec("fall", "FALL", confirm_seconds=0.4, reset_without_region=False,
                 description="lying posture from current-frame pose"),
    BehaviorSpec("fast_running", "RUNNING", confirm_seconds=0.8, blocks_small_person_display=False,
                 description="per-person running confirmation"),
)
CORE_NAMES = tuple(spec.name for spec in CORE_BEHAVIORS)

_registry = {spec.name: spec for spec in CORE_BEHAVIORS}


def register(spec):
    """Add an extension behaviour. Core entries can never be replaced."""
    if spec.kind != "extension":
        raise ValueError("Only extension behaviours can be registered at runtime")
    if spec.name in CORE_NAMES:
        raise ValueError(f"'{spec.name}' is a core behaviour and cannot be overridden")
    _registry[spec.name] = spec
    return spec


def unregister(name):
    if name in CORE_NAMES:
        raise ValueError("Core behaviours cannot be removed")
    _registry.pop(name, None)


def get(name):
    return _registry.get(name)


def all_behaviors():
    """Core first (fixed order), then extensions in registration order."""
    return tuple(_registry.values())


def extension_behaviors():
    return tuple(spec for spec in _registry.values() if spec.kind == "extension")


def display_label(name):
    spec = _registry.get(name)
    return spec.display_label if spec else str(name).upper()


def recorded_names():
    return tuple(_registry)


def reset_without_region_names():
    return tuple(spec.name for spec in CORE_BEHAVIORS if spec.reset_without_region)


def small_person_blocking_names():
    return tuple(spec.name for spec in CORE_BEHAVIORS if spec.blocks_small_person_display)


def gate_timing(name):
    """(confirm_seconds, release_seconds) for a gated behaviour."""
    spec = _registry.get(name)
    if spec is None:
        return 0.4, 0.25
    return spec.confirm_seconds, spec.release_seconds


def describe():
    return [spec.to_dict() for spec in _registry.values()]
