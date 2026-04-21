from __future__ import annotations

from . import sim_file_layout, xperience_raw


_ADAPTER_MODULES = (
    sim_file_layout,
    xperience_raw,
)

ADAPTER_REGISTRY = {module.ADAPTER_NAME: module for module in _ADAPTER_MODULES}
ADAPTER_NAMES = tuple(ADAPTER_REGISTRY.keys())


def get_adapter(name: str):
    try:
        return ADAPTER_REGISTRY[str(name)]
    except KeyError as exc:
        known = ", ".join(ADAPTER_NAMES)
        raise KeyError(f"Unknown dataset adapter {name!r}. Known adapters: {known}") from exc
