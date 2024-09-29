import sys, importlib.util
from amanda.compiler.transpile import GenOut
from amanda.config import STD_LIB
import amanda.runtime.py.rtbuiltins as ama_rt


def run(module: GenOut):
    mod_name = module.module.get_pymodule_name()
    bytecode = compile(module.py_code, mod_name, "exec")
    imports = resolve_imports(module)
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


def sort_imports(imports: list[GenOut]):
    resolved = set()
    unresolved = set([mod.module.fpath for mod in imports])
    new_imports = []

    while len(unresolved) != 0:
        for mod in imports:
            key = mod.module.fpath
            if key in resolved:
                continue
            if len(mod.module.imports) == 0:
                unresolved.remove(key)
                resolved.add(key)
                new_imports.append(mod)
            else:
                deps_loaded = True
                for imp in mod.module.imports:
                    deps_loaded = deps_loaded and imp in resolved
                if deps_loaded:
                    resolved.add(key)
                    unresolved.remove(key)
                    new_imports.append(mod)
    return new_imports


def resolve_imports(out: GenOut):
    imports = {}
    declare_builtins(imports)
    for mod_import in sort_imports(out.imports):
        module = mod_import.module
        mod_name = module.get_pymodule_name()
        if module.builtin:
            # Python source of a builtin module should be next to ama source
            mod_path = module.fpath.replace(".ama", ".py")
            py_mod = import_module_from_string(mod_name, read_file(mod_path))
        else:
            py_mod = import_module_from_string(
                mod_name, mod_import.py_code, imports
            )
        imports[mod_name] = py_mod
    return imports


def import_module_from_string(name: str, source: str, imports=None):
    if not imports:
        imports = {}
    spec = importlib.util.spec_from_loader(name, None)
    module = importlib.util.module_from_spec(spec)
    for imp, val in imports.items():
        module.__dict__[imp] = val
    bytecode = compile(source, name, "exec")
    exec(bytecode, module.__dict__)
    return module
