from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from os import path


@dataclass
class Module:
    fpath: str
    ast: Any = None
    loaded: bool = False
    builtin: bool = False
    compiled: bool = False
    imports: dict[str, Module] = field(default_factory=dict)

    def get_module_name(self):
        return path.basename(self.fpath).replace(".ama", "")

    def get_pymodule_name(self):
        return path.basename(self.fpath).replace(".", "_")

    def __str__(self) -> str:
        _, tail = path.split(self.fpath)
        return f"{tail}"
