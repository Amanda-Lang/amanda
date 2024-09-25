from __future__ import annotations
from dataclasses import dataclass

from utils.tycheck import unreachable


@dataclass
class Output:
    pass


@dataclass
class Str(Output):
    value: str

    def __str__(self):
        return self.value


@dataclass
class Group(Output):
    group: list[Output]


@dataclass
class Empty(Output):
    pass


@dataclass
class ArgsList(Output):
    elements: list[Output]


@dataclass
class Indented(Output):
    level: int
    inner: Output


@dataclass
class Lines(Output):
    n: int


def into_output(obj) -> Output:
    match obj:
        case str() | int() | float():
            if obj == "\n":
                return Lines(1)
            else:
                return Str(str(obj))
        case Output():
            return obj
        case list(objects):
            return Group([into_output(o) for o in objects])
        case _:
            unreachable("Invalid object")


def line() -> Lines:
    return Lines(1)
