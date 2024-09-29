from __future__ import annotations
from dataclasses import dataclass
from io import StringIO

from utils.tycheck import unreachable


INDENT = "    "


@dataclass
class OutputCtx:
    depth: int
    is_line_start: bool


@dataclass
class Output:
    def write(self, buff: StringIO, ctx: OutputCtx):
        raise NotImplementedError("Method must be overriden in subclass")


indent_spacing = lambda n: INDENT * n


@dataclass
class Str(Output):
    value: str
    ws: bool = True

    def write(self, buff: StringIO, ctx: OutputCtx):
        if ctx.is_line_start and ctx.depth > 0:
            buff.write(indent_spacing(ctx.depth))
            ctx.is_line_start = False
        buff.write(self.value.strip())
        if self.ws:
            buff.write(" ")

    def __str__(self):
        return self.value


@dataclass
class Group(Output):
    group: list[Output]

    def write(self, buff: StringIO, ctx: OutputCtx):
        for out in self.group:
            if not out:
                continue
            out.write(buff, ctx)


@dataclass
class Empty(Output):
    def write(self, buff: StringIO, ctx: OutputCtx):
        pass


@dataclass
class ArgsList(Output):
    elements: list[Output]

    def write(self, buff: StringIO, ctx: OutputCtx):
        if len(self.elements) == 0:
            return
        last = id(self.elements[-1])
        for out in self.elements:
            if not out:
                continue
            out.write(buff, ctx)
            if id(out) != last:
                buff.write(", ")


@dataclass
class Indented(Output):
    level: int
    inner: Output

    def write(self, buff: StringIO, ctx: OutputCtx):
        old_depth = ctx.depth
        ctx.depth += 1
        self.inner.write(buff, ctx)
        ctx.depth -= 1
        ctx.depth = old_depth


@dataclass
class Lines(Output):
    n: int

    def write(self, buff: StringIO, ctx: OutputCtx):
        buff.write("\n" * self.n)
        ctx.is_line_start = True


def into_output(obj, ws: bool = True) -> Output:
    match obj:
        case str() | int() | float():
            if obj == "\n":
                return Lines(1)
            else:
                return Str(str(obj), ws)
        case Output():
            return obj
        case list(objects):
            return Group([into_output(o, ws) for o in objects])
        case _:
            unreachable("Invalid object")


def line() -> Lines:
    return Lines(1)


def get_src(output: Output) -> str:
    buff = StringIO()
    output.write(buff, OutputCtx(-1, True))
    return buff.getvalue()
