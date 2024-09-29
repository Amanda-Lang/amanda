import sys, importlib.util
from amanda.compiler.transpile import GenOut
from amanda.config import STD_LIB
import amanda.runtime.py.rtbuiltins as ama_rt


def run(module: GenOut):
    mod_name = module.module.get_pymodule_name()
    bytecode = compile(module.py_code, mod_name, "exec")
    imports = resolve_imports(module)
    declare_builtins(imports)
    exec(bytecode, imports)


def read_file(fpath: str):
    with open(fpath, "r", encoding="utf8") as f:
        return f.read()


def declare_builtins(imports: dict):
    # verdadeiro e falso
    imports["verdadeiro"] = True
    imports["falso"] = False
    # Rt builtins
    imports["ama_rt"] = ama_rt


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
