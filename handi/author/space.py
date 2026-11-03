"""Named slots in the residual stream.

Every design allocates named sub-spaces (one-hot groups or scalars) of the residual stream and
builds matrices by referring to slots, never to raw dimension indices. All numbers in a weight
matrix therefore trace to a design decision in code, never to a typed literal or to training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Slot:
    name: str
    offset: int
    dim: int

    def index(self, i: int = 0) -> int:
        if not 0 <= i < self.dim:
            raise IndexError(f"slot {self.name} has dim {self.dim}, asked for {i}")
        return self.offset + i

    @property
    def indices(self) -> range:
        return range(self.offset, self.offset + self.dim)


class Space:
    """Allocator for named slots inside a vector of size `d`."""

    def __init__(self, d: int):
        self.d = d
        self.slots: dict[str, Slot] = {}
        self._next = 0

    def alloc(self, name: str, dim: int = 1) -> Slot:
        if name in self.slots:
            raise ValueError(f"slot {name!r} already allocated")
        if self._next + dim > self.d:
            raise ValueError(f"space of size {self.d} is full: cannot allocate {name!r} ({dim})")
        s = Slot(name, self._next, dim)
        self.slots[name] = s
        self._next += dim
        return s

    def __getitem__(self, name: str) -> Slot:
        return self.slots[name]

    @property
    def used(self) -> int:
        return self._next

    def vec(self, entries: dict[tuple[Slot, int] | Slot, float] | None = None) -> torch.Tensor:
        """A residual-space vector with the given (slot, index) -> value entries."""
        v = torch.zeros(self.d, dtype=torch.float32)
        for key, val in (entries or {}).items():
            slot, i = (key, 0) if isinstance(key, Slot) else key
            v[slot.index(i)] = val
        return v

    def basis(self, slot: Slot, i: int = 0) -> torch.Tensor:
        return self.vec({(slot, i): 1.0})

    def describe(self) -> str:
        return "\n".join(f"{s.offset:3d}..{s.offset + s.dim - 1:3d}  {s.name} [{s.dim}]" for s in self.slots.values())
