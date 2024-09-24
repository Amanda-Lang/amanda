import argparse
import time
from io import StringIO
import os
import sys
import subprocess
from os import path
from amanda.compiler.symbols.core import Module
from amanda.compiler.error import AmandaError, handle_exception, throw_error
from amanda.compiler.parse import parse
from amanda.compiler.transpile import transpile
from amanda.compiler.check.core import Analyzer
from amanda.compiler.codegen import ByteGen
from amanda.libamanda import run_module
import amanda.runtime


def write_file(name, code):
    with open(name, "w") as output:
        output.write(code)


def run_frontend(filename) -> tuple:
    try:
        program = parse(filename)
        valid_program = Analyzer(filename, [], Module(filename)).visit_module(
            program
        )
        return valid_program
    except AmandaError as e:
        throw_error(e)


def run_file(args):
    module, imports = run_frontend(args.file)
    out = transpile(module, imports)
    if args.debug:
        write_file("debug.py", out.py_code)
    amanda.runtime.run(out)


def main(*args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--debug", help="Generate a debug amasm file", action="store_true"
    )

    parser.add_argument("file", help="source file to be executed")

    if len(args):
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    if not path.isfile(args.file):
        sys.exit(
            f"The file '{path.abspath(args.file)}' was not found on this system"
        )
    run_file(args)


if __name__ == "__main__":
    main()
