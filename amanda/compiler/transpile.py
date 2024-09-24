from __future__ import annotations
from dataclasses import dataclass
from os import path
import sys
import pdb
from typing import cast
from io import StringIO
from amanda.compiler.check.exhaustiveness import (
    Case,
    DSuccess,
    DSwitch,
    Decision,
)
from amanda.compiler.module import Module
import amanda.compiler.symbols.core as symbols
from amanda.compiler.symbols.base import Constructor, Symbol
from amanda.compiler.types.core import (
    IntCons,
    StrCons,
    Type,
    Types,
    Primitive,
    Uniao,
    Variant,
    VariantCons,
)
import amanda.compiler.ast as ast
from amanda.compiler.tokens import TokenType as TT
from amanda.compiler.error import AmandaError, throw_error
from amanda.compiler.types.builtins import Builtins
from utils.tycheck import unreachable, unwrap


@dataclass
class GenOut:
    module: Module
    py_code: str
    src_map: dict[int, int]
    imports: list[GenOut]


class PyGen:
    INDENT = "    "

    def __init__(self, module: Module):
        self.py_lineno = 0  # tracks lineno in compiled python src
        self.ama_lineno = 1  # tracks lineno in input amanda src
        self.depth = -1  # current indent level
        self.func_depth = 0
        self.class_depth = 0
        self.program_symtab = None
        self.scope_symtab: symbols.Scope
        self.uniao_variants: dict[str, int] = {}
        self.ctx_module: Module = module
        self.src_map = {}  # Maps py_fileno to ama_fileno
        self.escolha_ret_var: str = "_re0_"
        self.modules = {}
        self.module = module
        self.prepend_buffer: StringIO | None = None
        self.buffer = StringIO()

    def transpile(self, imports: dict[str, Module]) -> GenOut:
        """Method that begins compilation of amanda source."""
        program = self.ctx_module.ast
        self.program_symtab = self.scope_symtab = program.symbols

        if self.ctx_module.builtin:
            return self.transpile_builtin()

        for mod in imports.values():
            self.modules[mod.fpath] = mod.get_pymodule_name()

        compiled_imports = []
        for mod in imports.values():
            if mod.fpath in self.modules and mod.compiled:
                continue
            compiler = PyGen(mod)
            compiler.modules = self.modules
            module_out = compiler.transpile({})
            compiled_imports.append(module_out)
            mod.compiled = True

        import_strs = []  # self.gen_imports(self.module.imports)
        py_code = self.compile_block(program, import_strs)
        return GenOut(
            self.module,
            py_code,
            self.src_map,
            compiled_imports,
        )

    def gen_imports(self, imports: dict[str, Module]) -> list[str]:
        import_strs = []
        for mod in imports.values():
            mod_name = mod.get_pymodule_name()
            import_strs.append(f"import {mod_name}")
        return import_strs

    def get_module_name(self, fpath: str):
        mod_name = path.basename(fpath).replace(".", "_")
        if mod_name in self.modules:
            mod_name = f"{mod_name}{len(self.modules)}"
        return mod_name

    def transpile_builtin(self) -> GenOut:
        return GenOut(
            self.module,
            "",
            {},
            [],
        )

    def bad_gen(self, node):
        raise NotImplementedError(
            f"Cannot generate code for this node type yet: (TYPE) {type(node)}"
        )

    def gen(self, node: ast.ASTNode):
        node_class = type(node).__name__.lower()
        method_name = f"gen_{node_class}"
        gen_method = getattr(self, method_name, self.bad_gen)
        # Update line only if node type has line attribute
        self.ama_lineno = getattr(node, "lineno", self.ama_lineno)
        if node_class == "block":
            return gen_method(node, args)
        return gen_method(node)

    def gen_program(self, node):
        return self.compile_block(node, [])

    def gen_usa(self, node: ast.Usa):
        module = node.items
        generator = PyGen(module)
        mod_src, _ = generator.generate_code(
            module,
        )
        self.py_lineno += generator.py_lineno
        return mod_src

    def update_line_info(self):
        self.src_map[self.py_lineno] = self.ama_lineno
        self.py_lineno += 1

    def build_str(self, str_buffer: StringIO):
        string = str_buffer.getvalue()
        str_buffer.close()
        return string

    def writeln_to_buff(self, buffer: StringIO, py: str):
        indent_level = self.INDENT * self.depth
        buffer.write(f"{indent_level}{py}\n")

    def write_py(self, py: str, indent=False):
        indent_level = self.INDENT * self.depth if indent else ""
        self.buffer.write(f"{indent_level}{py}")

    def writeln_py(self, py: str, indent=False):
        indent_level = self.INDENT * self.depth if indent else ""
        self.buffer.write(f"{indent_level}{py}\n")
        self.update_line_info()

    def enter_scope(self):
        self.depth += 1

    def leave_scope(self):
        if self.depth == 0:
            unreachable("Cannot leave global scope")
        self.depth -= 1

    def compile_block(self, node: ast.Block, stmts: list[str]):
        # stmts param is a list of stmts
        # node defined here because caller may want
        # to add custom statement to the beginning of
        # a block
        self.depth += 1
        self.scope_symtab = node.symbols
        # Newline for header
        self.update_line_info()
        block = StringIO()
        # Add caller stmts to buffer
        for stmt in stmts:
            block.write(f"{indent_level}{stmt}")
            block.write("\n")
            self.update_line_info()

        for child in node.children:
            self.gen(child)
        self.depth -= 1
        self.scope_symtab = self.scope_symtab.enclosing_scope

    def gen_vardecl(self, node):
        assign = node.assign
        name = node.name.lexeme
        symbol = unwrap(self.scope_symtab.resolve(name))
        self.write_py(f"{symbol.out_id} = ", indent=True)
        if assign:
            self.gen(assign.right)
        else:
            # Check for initializer
            init_values = {
                "int": 0,
                "real": 0.0,
                "bool": "falso",
                "texto": '''""''',
            }
            value = init_values.get(str(node.var_type))
            self.write_py(f"{value}")
        self.writeln_py("")

    def gen_uniao(self, node: ast.Uniao):
        name = node.name.lexeme
        sym: Uniao = cast(Uniao, self.scope_symtab.resolve_typed(name))
        for variant in sym.variants.values():
            self.get_variant_index(variant)

    def get_variant_index(self, variant: Variant) -> int:
        return self.uniao_variants.setdefault(
            variant.variant_id(), len(self.uniao_variants)
        )

    def gen_functiondecl(self, node):
        name = node.name.lexeme
        func_def = StringIO()
        func_symbol = self.scope_symtab.resolve(name)
        name = func_symbol.out_id
        params = []
        for param in func_symbol.params.values():
            params.append(param.out_id)
        params = ",".join(params)
        func_def.write(f"def {name}({params}):\n")
        self.func_depth += 1
        func_def.write(
            self.get_func_block(node.block),
        )
        self.func_depth -= 1
        return self.build_str(func_def)

    def gen_classdecl(self, node):
        name = node.name.lexeme
        class_def = StringIO()
        klass = self.scope_symtab.resolve(name)
        name = klass.out_id
        # name of base class for user defined types
        base_class = "_BaseClass_"
        class_def.write(f"class {name}(_BaseClass_):\n")
        self.class_depth += 1
        class_def.write(self.compile_block(node.body, []))
        self.class_depth -= 1
        return self.build_str(class_def)

    def get_func_block(self, block):
        # Add global and nonlocal statements
        # to beginning of a function
        stmts = []
        scope = block.symbols
        if self.func_depth >= 1:
            global_stmt = self.gen_global_stmt()
            if global_stmt:
                stmts.append(global_stmt)
        if self.func_depth > 1:
            non_local = self.gen_nonlocal_stmt(scope)
            if non_local:
                stmts.append(non_local)
        return self.compile_block(block, stmts)

    def get_names(self, scope, include_funcs=False):
        names = []
        for symbol in scope.symbols.values():
            symbol_type = type(symbol)

            if symbol_type == symbols.VariableSymbol or (
                include_funcs and symbol_type == symbols.FunctionSymbol
            ):
                names.append(symbol.out_id)

        return names

    # TODO: Fix unecessary forward global declarations
    def gen_global_stmt(self):
        """Adds global statements to
        the beginning of a function"""
        names = self.get_names(self.program_symtab)
        if len(names) == 0:
            return None
        names = ",".join(names)
        return f"global {names}"

    def gen_nonlocal_stmt(self, scope):
        """Adds nonlocal statements to
        the beginning of a function"""
        names = []
        scope = scope.enclosing_scope
        while scope and scope.enclosing_scope is not None:
            names += self.get_names(scope)
            scope = scope.enclosing_scope
        if len(names) == 0:
            return None
        names = ",".join(names)
        return f"nonlocal {names}"

    def gen_listliteral(self, node):
        elements = ",".join(
            [str(self.gen(element)) for element in node.elements]
        )
        list_type = node.eval_type.get_type()
        return f"Lista({list_type},[{elements}])"

    def gen_call(self, node):
        func = node.symbol
        args = ",".join([str(self.gen(arg)) for arg in node.fargs])
        callee = self.gen(node.callee)
        if isinstance(func, Variant):
            return f"ama_rt.build_variant({callee}, {args})"
        elif func.is_type():
            unreachable("Have not yet implemented calling types!!!")
        func_call = f"{callee}({args})"
        return self.gen_expression(func_call, node.prom_type)

    def gen_alvo(self, node):
        return "eu"

    def gen_set(self, node):
        target = self.gen(node.target)
        expr = self.gen(node.expr)
        return f"{target} = {expr}"

    def gen_index(self, node):
        target = self.gen(node.target)
        index = self.gen(node.index)
        index_expr = f"{target}[{index}]"
        return self.gen_expression(index_expr, node.prom_type)

    def gen_get(self, node):
        target = self.gen(node.target)
        member = node.member.lexeme
        klass = node.target.eval_type
        member_sym = klass.members.get(member)
        get_expr = f"{target}.{member_sym.out_id}"
        return self.gen_expression(get_expr, node.prom_type)

    def gen_assign(self, node):
        lhs = self.gen(node.left)
        rhs = self.gen(node.right)
        return f"{lhs} = {rhs}"

    def gen_constant(self, node):
        prom_type = node.prom_type
        literal = str(node.token.lexeme)
        return self.gen_expression(literal, prom_type)

    def gen_fmtstr(self, node):
        return "".join([self.gen(part) for part in node.parts])

    def gen_indexget(self, node, gen_get=True):
        target = self.gen(node.target)
        index = self.gen(node.index)
        return f"{target}[{index}]"

    def gen_indexset(self, node):
        index_get = self.gen_indexget(node.index, gen_get=False)
        new_val = self.gen(node.value)
        return f"{index_get} = {new_val}"

    def load_variable(self, symbol: Symbol) -> str:
        name = symbol.out_id
        sym_module = cast(symbols.Typed, symbol).module.fpath
        typed_sym = cast(symbols.Typed, symbol)
        # Guarantee item is in the name table
        # self.get_table_index(name, self.NAME_TABLE)
        if symbol.is_external(self.ctx_module):
            mod_id = self.modules[sym_module]
            return f"{mod_id}.{name}"
        return symbol.out_id

    def gen_variable(self, node):
        name = node.token.lexeme
        symbol = node.var_symbol
        # TODO: Make sure that every identifier goes through
        # 'visit_variable' so that symbol attribute can be set
        if symbol is None:
            symbol = self.scope_symtab.resolve(name)
        expr = self.load_variable(symbol)
        prom_type = node.prom_type
        return self.gen_expression(expr, prom_type)

    def gen_binop(self, node):
        lhs = self.gen(node.left)
        rhs = self.gen(node.right)
        operator = node.token.lexeme
        if operator == "e":
            operator = "and"
        elif operator == "ou":
            operator = "or"
        binop = f"({lhs} {operator} {rhs})"
        # Promote node
        return self.gen_expression(binop, node.prom_type)

    def gen_unaryop(self, node):
        operator = node.token.lexeme
        operand = self.gen(node.operand)
        if operator == "nao":
            operator = "not"
        unaryop = f"({operator} {operand})"
        return self.gen_expression(unaryop, node.prom_type)

    def gen_converta(self, node: ast.Converta):
        target_t = node.target.eval_type
        new_t = node.eval_type
        # Converting to same type, can ignore this
        # TODO: Do this in sem analysis
        if new_t == target_t or new_t == Builtins.Indef:
            return ""
        expr = self.gen(node.target)
        ty = self.load_variable(new_t)
        return f"ama_rt.converta({expr}, {ty})"

    def gen_expression(self, expression, prom_type: Type):
        match prom_type:
            case Primitive(tag=tag) if tag == Types.TREAL:
                return f"float({expression})"
            case _:
                return expression

    def gen_senaose(self, node):
        senaose_stmt = StringIO()
        condition = self.gen(node.condition)
        then_branch = self.compile_branch(node.then_branch)
        indent_level = self.INDENT * self.depth
        senaose_stmt.write(f"{indent_level}elif {condition}:\n{then_branch}")
        return self.build_str(senaose_stmt)

    def gen_se(self, node):
        if_stmt = StringIO()
        condition = self.gen(node.condition)
        then_branch = self.compile_branch(node.then_branch)
        elsif_branches = "".join(
            [self.gen(branch) for branch in node.elsif_branches]
        )
        if_stmt.write(f"if {condition}:\n{then_branch}")
        if_stmt.write(f"{elsif_branches}")
        if node.else_branch:
            indent_level = self.INDENT * self.depth
            else_branch = self.compile_branch(node.else_branch)
            if_stmt.write(f"\n{indent_level}else:\n{else_branch}")
        return self.build_str(if_stmt)

    def compile_branch(self, block):
        scope = block.symbols
        branch = self.compile_block(block, [])
        names = self.get_names(scope, include_funcs=True)
        if len(names) > 0:
            self.update_line_info()
            names = ",".join(names)
            indent_level = self.INDENT * (self.depth + 1)
            branch += f"{indent_level}del {names}"
        return branch

    def gen_enquanto(self, node):
        condition = self.gen(node.condition)
        body = self.compile_branch(node.statement)
        return f"while {condition}:\n{body}"

    def gen_module(self, node: ast.Module):
        return self.compile_block(node, [])

    def gen_iguala(self, node: ast.Iguala):

        buff = StringIO()
        self.gen_iguala_from_ir(node, node.ir.tree, buff)
        # print(buff.getvalue())
        self.prepend_buffer = buff
        return self.escolha_ret_var

    def gen_path(self, node: ast.Path):
        symbol = node.symbol
        if not symbol:
            raise ValueError("Symbol should have been resolved before codegen!")
        match symbol:
            case Variant():
                tag = self.uniao_variants[symbol.variant_id()]
                if not symbol.is_callable():
                    # No args means we can generate the code for creating the variant here
                    return f"ama_rt.build_variant({tag})"
                else:
                    return f"{tag}"
            case _:
                raise NotImplementedError("Cannot generate code for symbol")

    def gen_iguala_from_ir(
        self, node: ast.Iguala, decision: Decision, buff: StringIO
    ):
        match decision:
            case DSuccess(body):
                bindings = []
                for name, variable in body.bindings:
                    var = self.load_variable(variable)
                    source = unwrap(unwrap(body.value.symbols).resolve(name))
                    bindings.append(f"{name} = {source.out_id}")
                body = self.compile_block(body.value, bindings)
                buff.write(f"{body}\n")
            case DSwitch(variable=var, cases=cases):
                self.gen_iguala_cases(node, var, cases, buff)
            case _:
                unreachable("Invalid decision node")

    def gen_iguala_test(
        self, var: symbols.VariableSymbol, constructor: Constructor
    ):
        test_var = self.load_variable(var)
        match constructor:
            case IntCons(val) | StrCons(val):
                return f"{test_var} ==  {val}"
            case VariantCons(tag=tag, uniao=uniao):
                variant = uniao.variant_by_tag(tag)
                # TODO: Fix this please!!!
                idx = self.uniao_variants[variant.variant_id()]
                return f"{idx} == {test_var}.tag"
            case _:
                unreachable("Unknown constructor")

    def gen_iguala_case_bindings(
        self, var: symbols.VariableSymbol, case: Case, buff: StringIO
    ):
        self.load_variable(var)
        variant = var.out_id
        match case.constructor:
            case VariantCons(name=name):
                cargs = case.arguments
                if not cargs:
                    return ""
                for i, arg in enumerate(cargs):
                    self.writeln_to_buff(
                        buff, f"{arg.out_id} = {variant}.args[{i}]"
                    )
            case _:
                return ""

    def gen_iguala_cases(
        self,
        node: ast.Iguala,
        var: symbols.VariableSymbol,
        cases: list[Case],
        buff: StringIO,
    ):
        case = cases[0]
        test = self.gen_iguala_test(var, case.constructor)
        print("depth: ", self.depth)
        self.writeln_to_buff(buff, f"if {test}:")
        self.enter_scope()
        self.gen_iguala_case_bindings(var, case, buff)
        self.leave_scope()
        self.gen_iguala_from_ir(node, case.body, buff)

        for case in cases[1:]:
            test = self.gen_iguala_test(var, case.constructor)
            self.writeln_to_buff(buff, f"elif {test}:")
            self.enter_scope()
            self.gen_iguala_case_bindings(var, case, buff)
            self.leave_scope()
            self.gen_iguala_from_ir(node, case.body, buff)

    def gen_escolha(self, node):
        switch = StringIO()
        expr = self.gen(node.expression)
        cases = node.cases
        default = node.default_case
        indent = self.INDENT * self.depth
        # No need to generate code
        if not len(cases) and not default:
            return ""
        for c, case in enumerate(cases):
            value = self.gen(case.expression)
            block = self.compile_block(case.block, [])
            if c == 0:
                switch.write(f"if {expr} == {value}:\n{block}")
            else:
                switch.write(f"{indent}elif {expr} == {value}:\n{block}")
        if default:
            block = self.compile_block(default, [])
            if not len(cases):
                switch.write(f"if True:\n{block}")
            else:
                switch.write(f"{indent}else:\n{block}")
        return self.build_str(switch)

    def gen_para(self, node):
        scope = node.statement.symbols
        # Change control var name to local name
        para_expr = node.expression
        control_var = scope.resolve(para_expr.name.lexeme)
        body = self.compile_branch(node.statement)
        expression = self.gen_paraexpr(para_expr, control_var)
        return f"for {expression}:\n{body}"

    def gen_paraexpr(self, node, control_var):
        range_expr = node.range_expr
        name = control_var.out_id
        start = self.gen(range_expr.start)
        stop = self.gen(range_expr.end)
        inc = f"-1 if {start} > {stop} else 1"
        if range_expr.inc:
            inc = self.gen(range_expr.inc)
        return f"{name} in range({start},{stop},{inc})"

    def gen_retorna(self, node):
        expression = self.gen(node.exp) if node.exp else "None"
        return f"return {expression}"

    def gen_produz(self, node: ast.Produz):
        if not node.exp or node.exp.eval_type in (
            Builtins.Unknown,
            Builtins.Vazio,
        ):
            return ""
        expression = self.gen(node.exp)
        return f"{self.escolha_ret_var} = {expression}"

    def gen_mostra(self, node):
        expression = self.gen(node.exp)
        if node.exp.eval_type.otype == Types.TVAZIO:
            expression = "vazio"
        return f"printc({expression})"

    def gen_loopctlstmt(self, node):
        token = node.token
        return "break" if token.token == TT.QUEBRA else "continue"


def transpile(module: Module, imports: dict[str, Module]) -> GenOut:
    transpiler = PyGen(module)
    return transpiler.transpile(imports)
