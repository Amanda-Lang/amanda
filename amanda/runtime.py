import sys, importlib.util
from amanda.compiler.transpile import GenOut
from amanda.config import STD_LIB


def run(module: GenOut):
    bytecode = compile(module.py_code, "<string>", "exec")
    imports = resolve_imports(module)
    exec(bytecode, imports)


def read_file(fpath: str):
    with open(fpath, "r", encoding="utf8") as f:
        return f.read()


def resolve_imports(out: GenOut):
    imports = {}
    for mod_import in out.imports:
        module = mod_import.module
        if module.builtin:
            # Python source of a builtin module should be next to ama source
            mod_path = module.fpath.replace(".ama", ".py")
            mod_name = module.get_pymodule_name()
            py_mod = import_module_from_string(mod_name, read_file(mod_path))
            imports[mod_name] = py_mod
        else:
            pass
    return imports


def import_module_from_string(name: str, source: str):
    spec = importlib.util.spec_from_loader(name, None)
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)
    return module
