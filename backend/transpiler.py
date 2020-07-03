'''
Module responsible for generating python code
from an AST
'''
from io import StringIO
import amanda.symbols as symbols
import amanda.ast_nodes as ast
import amanda.semantic as sem
import amanda.error as error
from amanda.parser import Parser
import backend.generators as generators

class Transpiler:

    def __init__(self,src,debug=False):
        self.src = src
        self.handler = error.ErrorHandler.get_handler()
        self.compiled_src = None
        #used to set indentation in blocks
        self.indent_level = 0
        self.debug = debug
        self.test_buffer = None
        if self.debug:
            #If debug is enabled, redirect output to
            #an in memory buffer
            self.test_buffer = StringIO()


    def compile(self):
        ''' Method that runs an Amanda script. Errors raised by the
        frontend are handled by an error handler instance'''
        try:
            program = Parser(self.src).parse()
            valid_program = sem.Analyzer().check_program(program)
            code_objs = self.gen(valid_program)
        except error.AmandaError as e:
            if self.debug:
                self.test_buffer.write(str(e).strip())
                sys.exit()
            self.handler.throw_error(e,self.src)

        str_buffer = StringIO()

        for obj in code_objs:
            print(obj,end="\n\n",file=str_buffer)

        self.compiled_src = str_buffer.getvalue()
        return self.compiled_src

    def exec(self):
        if not self.compiled_src:
            self.compile()
        code = compile(self.compiled_src,"<string>","exec")
        scope = {}
        if self.debug:
        #Add buffer to local dict
            scope["__buffer__"] = self.test_buffer
        exec(code,scope)



    def bad_gen(self,node):
        raise NotImplementedError(f"Cannot generate code for this node type yet: (TYPE) {type(node)}")

    def gen(self,node,args=None):
        node_class = type(node).__name__.lower()
        method_name = f"gen_{node_class}"
        gen_method = getattr(self,method_name,self.bad_gen)
        return gen_method(node)


    def gen_program(self,node):
        code_objs = []
        for child in node.children:
            code_objs.append(self.gen(child))
        return code_objs
            
    
    def gen_block(self,node,scope=None):
        self.indent_level += 1
        gens = []
        for child in node.children:
            gens.append(self.gen(child))
        level = self.indent_level
        self.indent_level -= 1
        return generators.Block(gens,level)

    def gen_vardecl(self,node):
        assign = node.assign
        if assign:
            return self.gen(assign)
        return generators.VarDecl(node.name.lexeme,node.var_type.tag)

    def gen_functiondecl(self,node):
        params = []
        for param in node.params:
            params.append(param.name.lexeme)
        return generators.FunctionDecl(
            node.name.lexeme,
            self.gen(node.block),
            params
        )

    def gen_call(self,node) :
        args = [self.gen(arg) for arg in node.fargs]
        return generators.Call(self.gen(node.callee),args)


    def gen_assign(self,node):
        lhs = self.gen(node.left)
        rhs = self.gen(node.right)
        return generators.Assign(lhs,rhs)

    
    def gen_constant(self,node):
        return node.token.lexeme

    def gen_variable(self,node):
        return node.token.lexeme

    def gen_binop(self,node):
        lhs = self.gen(node.left)
        rhs = self.gen(node.right)
        return generators.BinOp(node.token.lexeme,lhs,rhs)

    def gen_unaryop(self,node):
        return generators.UnaryOp(
            node.token.lexeme,
            self.gen(node.operand)
        )

    
    def gen_se(self,node):

        gen = generators.Se(
            self.gen(node.condition),
            self.gen(node.then_branch)
        )

        if node.else_branch:
            gen.else_branch = generators.Senao(
                self.gen(node.else_branch),
                self.indent_level
            )
        return gen

    def gen_enquanto(self,node):
        return generators.Enquanto(
            self.gen(node.condition),
            self.gen(node.statement)
        )


    def gen_para(self,node):
        return generators.Para(
            self.gen(node.expression),
            self.gen(node.statement)
        )

    def gen_paraexpr(self,node):
        range_expr = node.range_expr
        gen = generators.ParaExpr(
            node.name.lexeme,
            self.gen(range_expr.start),
            self.gen(range_expr.end),
        )
        if range_expr.inc:
            gen.inc = self.gen(range_expr.inc)
        return gen
            

    def gen_retorna(self,node):
        return generators.Retorna(self.gen(node.exp))

    def gen_mostra(self,node):
        expression = self.gen(node.exp)
        return generators.Mostra(expression,self.debug)






