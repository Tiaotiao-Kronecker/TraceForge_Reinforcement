from __future__ import annotations


def retired_entrypoint_message(*, entrypoint: str, replacement: str, reason: str) -> str:
    return (
        f"{entrypoint} 已下线。\n"
        f"原因: {reason}\n"
        f"请改用: {replacement}"
    )


def exit_retired_entrypoint(*, entrypoint: str, replacement: str, reason: str) -> None:
    raise SystemExit(
        retired_entrypoint_message(
            entrypoint=entrypoint,
            replacement=replacement,
            reason=reason,
        )
    )
