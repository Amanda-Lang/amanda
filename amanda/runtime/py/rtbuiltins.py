from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Variant:
    tag: int
    args: list

    def __str__(self) -> str:
        return f"<Variante({self.tag})>"


@dataclass
class Registo:
    _data_: dict[str, Any]

    def __post_init__(self):
        self.__dict__ = self._data_


def lista(ty, elements):
    return elements


def converta(obj, ty):
    if ty in (int, float, str, bool):
        return ty(obj)
    else:
        return obj
