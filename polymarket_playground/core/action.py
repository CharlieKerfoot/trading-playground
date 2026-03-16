"""Action schema for trading decisions."""

from __future__ import annotations

from dataclasses import dataclass

# Maps string actions to direction integers
DIRECTION_MAP = {"hold": 0, "buy_yes": 1, "buy_no": 2, "close": 3}
DIRECTION_NAMES = {v: k for k, v in DIRECTION_MAP.items()}


@dataclass
class Action:
    """A trading action."""
    direction: int = 0  # 0=hold, 1=buy_yes, 2=buy_no, 3=close
    size: float = 0.0   # fraction of max position [0, 1]
    reasoning: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Action:
        direction = d.get("action", "hold")
        if isinstance(direction, str):
            direction = DIRECTION_MAP.get(direction, 0)
        return cls(
            direction=direction,
            size=float(d.get("size", 0.0)),
            reasoning=d.get("reasoning", ""),
        )

    @property
    def is_hold(self) -> bool:
        return self.direction == 0

    @property
    def direction_name(self) -> str:
        return DIRECTION_NAMES.get(self.direction, "hold")
