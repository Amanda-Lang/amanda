from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Variant:
    tag: int
    args: list

    def __str__(self) -> str:
        return f"<Variante({self.tag})>"


def lista(ty, elements):
    return elements


def converta(obj, ty):
    if ty in (int, float, str, bool):
        return ty(obj)
    else:
        return obj
