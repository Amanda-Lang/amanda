from enum import Enum
from amanda.lexer import Lexer
from amanda.tokens import TokenType as TT
from amanda.tokens import Token
import amanda.ast_nodes as ast
import amanda.symbols as SYM
import amanda.error as error
import amanda.natives as natives


#TODO: Refactor this because it doesn't scale
''' Class to represent built in types '''
class Type(Enum):
    INT = 0
    REAL = 1
    BOOL = 2
    
    def __str__(self):
        return self.name.lower()

# results of static type computation
# None means illegal operation
aritop_results = [
#       int       real       bool
    [Type.INT,Type.REAL,None],
    [Type.REAL,Type.REAL,None],
    [None,None,None],
]

#Table for  type for >,<,>=,<=
relop_results = [
#       int       real       bool
    [Type.BOOL,Type.BOOL,None],
    [Type.BOOL,Type.BOOL,None],
    [None,None,None],
]

#table for type results for == !=
eqop_results = [
#       int       real       bool
    [Type.BOOL,Type.BOOL,Type.BOOL],
    [Type.BOOL,Type.BOOL,Type.BOOL],
    [None,None,Type.BOOL],
]

#table for type results for e ou !
logop_results = [
#    int  real  bool
    [None,None,None],
    [None,None,None],
    [None,None,Type.BOOL],
]



# table for type promotions
# VAZIO means should not be promoted
type_promotion= [
#    int  real  bool
    [None,Type.REAL,None],
    [None,None,None],
    [None,None,None],
]


'''
Class that performs semantic analysis on a syntatically valid
amanda program.
'''

class Analyzer(ast.Visitor):

    def __init__(self,src):
        self.current_scope = SYM.Scope(SYM.Scope.GLOBAL)
        self.src = src 
        self.init_builtins()

    def init_builtins(self):
        self.current_scope.define("int",SYM.BuiltInType(Type.INT))
        self.current_scope.define("real",SYM.BuiltInType(Type.REAL))
        self.current_scope.define("bool",SYM.BuiltInType(Type.BOOL))
        builtins = natives.builtin_types.values()

        for type_obj in builtins:
            type_obj.load_symbol(self.current_scope)
        
        for type_obj in builtins:
            type_obj.define_symbol(self.current_scope)
        

    def has_return(self,node):
        node_class = type(node).__name__.lower()
        method_name = f"has_return_{node_class}"
        visitor_method = getattr(self,method_name,self.general_check)
        return visitor_method(node)

    def general_check(self,node):
        return False

    def has_return_block(self,node):
        for child in node.children:
            has_return = self.has_return(child)
            if has_return:
                return True
        return False

    def has_return_se(self,node):
        ''' Method checks for return within
            'se' statements'''
        # If there is no else branch return None immediately
        return False if not node.else_branch else self.has_return(node.else_branch)


    def has_return_retorna(self,node):
        return True


    def error(self,line,col,code,**kwargs):
        message = code.format(**kwargs)
        handler = error.ErrorHandler.get_handler()
        handler.throw_error(
            error.Analysis(message,line,col),
            self.src
        )


    def check_program(self,program):
        self.visit(program)
        return program

    def visit_program(self,node):
        for child in node.children:
            self.visit(child)

    def visit_vardecl(self,node):
        name = node.name.lexeme
        line = node.name.line
        col = node.name.col
        var_type = self.current_scope.resolve(node.var_type.lexeme)
        if not var_type or not var_type.is_type():
            self.error(
                        line,col,
                        error.Analysis.UNDEFINED_TYPE,
                        type=node.var_type.lexeme
                    )

        if not self.current_scope.symbols.get(name) is None:
            self.error(
                        line,col,
                        error.Analysis.ID_IN_USE,
                        name=name
                    )

        symbol = SYM.VariableSymbol(name,var_type)
        self.current_scope.define(name,symbol)
        if node.assign is not None:
            if node.assign.right.token.lexeme == name:
                self.error(
                            line,
                            f"Erro ao inicializar variável. Não pode referenciar uma variável durante a sua declaração")

            self.visit(node.assign)


    def visit_functiondecl(self,node):
        #Check if id is already in use
        name = node.name.lexeme
        line = node.name.line
        col = node.name.col
        if self.current_scope.resolve(name):
            self.error(line,error.Analysis.ID_IN_USE,name=name)
        #Check if return types exists
        function_type =  self.current_scope.resolve(node.func_type.lexeme)
        if not function_type or not function_type.is_type():
            self.error(line,col,error.Analysis.UNDEFINED_TYPE,type=node.func_type.lexeme)
        has_return = self.has_return(node.block)
        if not has_return:
            self.error(line,col,error.Analysis.NO_RETURN_STMT,name=name)
        params = {}
        for param in node.params:
            param_name = param.name.lexeme
            if params.get(param_name):
                self.error(line,col,error.Analysis.REPEAT_PARAM,name=param_name)
            param_symbol = self.visit(param)
            params[param_name] = param_symbol
            #Add params o current_scope
        symbol = SYM.FunctionSymbol(name,function_type,params)
        self.current_scope.define(name,symbol)
        scope = SYM.Scope(name,self.current_scope)
        for name,param in symbol.params.items():
            scope.define(name,param)
        self.visit(node.block,scope)

    def visit_block(self,node,scope=None):
        if not scope:
            scope = SYM.Scope(SYM.Scope.LOCAL,self.current_scope)
        self.current_scope = scope
        for child in node.children:
            self.visit(child)
        self.current_scope = self.current_scope.enclosing_scope


    def visit_param(self,node):
        var_type = self.current_scope.resolve(node.param_type.lexeme)
        if not var_type or not var_type.is_type():
            self.error(
                        node.param_type.line,
                        node.param_type.col,
                        error.Analysis.UNDEFINED_TYPE,
                        type=node.param_type.lexeme
                    )
        name = node.name.lexeme
        return SYM.VariableSymbol(name,var_type)


    def visit_constant(self,node):

        if node.token.token == TT.INTEGER:
            node.eval_type = Type.INT
        elif node.token.token == TT.REAL:
            node.eval_type = Type.REAL
        elif node.token.token == TT.STRING:
            raise Exception("Not implemented strings yet")
        elif node.token.token in (TT.VERDADEIRO,TT.FALSO):
            node.eval_type = Type.BOOL
    
    def visit_variable(self,node):
        name = node.token.lexeme
        line = node.token.line
        col = node.token.col
        sym = self.current_scope.resolve(name)
        if not sym:
            self.error(line,col,error.Analysis.UNDECLARED_ID,name=name)
        #Referencing array by name 
        elif not sym.is_valid_var():
            self.error(line,col,error.Analysis.INVALID_REF,name=name)
        node.eval_type = sym.type.name


    def visit_binop(self,node):
        self.visit(node.left)
        self.visit(node.right)
        #Evaluate type of binary
        #arithmetic operation
        operator = node.token.token
        if operator in (TT.PLUS,TT.MINUS,TT.STAR,TT.SLASH,TT.MODULO):
            node.eval_type = aritop_results[node.left.eval_type.value][node.right.eval_type.value]
        elif operator in (TT.GREATER,TT.LESS,TT.GREATEREQ,TT.LESSEQ):
            node.eval_type = relop_results[node.left.eval_type.value][node.right.eval_type.value]
        elif operator in (TT.DOUBLEEQUAL,TT.NOTEQUAL):
            node.eval_type = eqop_results[node.left.eval_type.value][node.right.eval_type.value]
        elif operator in (TT.AND,TT.OR):
            node.eval_type = logop_results[node.left.eval_type.value][node.right.eval_type.value]
        #Validate binary ops
        line = node.token.line
        col = node.token.col
        lexeme = node.token.lexeme
        if not node.eval_type:
            self.error(
                line,col,
                error.Analysis.INVALID_OP,
                t1=node.left.eval_type,
                t2=node.right.eval_type,
                operator=lexeme
            )

        node.left.prom_type = type_promotion[node.left.eval_type.value][node.right.eval_type.value]
        node.right.prom_type = type_promotion[node.right.eval_type.value][node.left.eval_type.value]

    def visit_unaryop(self,node):
        self.visit(node.operand)
        operator = node.token.token
        lexeme = node.token.lexeme
        line = node.token.line
        type = node.operand.eval_type
        if operator in (TT.PLUS,TT.MINUS):
            if type != Type.INT and type != Type.REAL:
                self.error(line,error.Analysis.INVALID_UOP,operator=lexeme,type=type)
            node.eval_type = type
        elif operator == TT.NAO:
            node.operand.prom_type = logop_results[type.value][Type.BOOL.value]
            if type != Type.BOOL and not node.operand.prom_type:
                self.error(line,error.Analysis.INVALID_UOP,operator=lexeme,type=type)
            node.eval_type = node.operand.prom_type


    def visit_assign(self,node):
        self.visit(node.right)
        self.visit(node.left)

        #Set node types
        node.eval_type = node.left.eval_type
        node.prom_type = None
        #Set promotion type for right side
        node.right.prom_type = type_promotion[node.right.eval_type.value][node.left.eval_type.value]

        line = node.token.line
        col = node.token.col
        if node.left.eval_type != node.right.eval_type and not node.right.prom_type:
            self.error(
                    line,col,
                    f"atribuição inválida. incompatibilidade entre os operandos da atribuição [{node.left.eval_type} = {node.right.eval_type}]")


    def visit_mostra(self,node):
        self.visit(node.exp)


    def visit_retorna(self,node):
        self.visit(node.exp)
        token = node.token.token
        line = node.token.line
        col = node.token.col
        function = self.current_scope.get_enclosing_func()
        #TODO: Fix return bug inside local scope
        if not function:
            self.error(line,col,f"O comando 'retorna' só pode ser usado dentro de uma função")
        function = self.current_scope.resolve(function.name)
        node.exp.prom_type = type_promotion[node.exp.eval_type.value][function.type.name.value]
        if not function.type:
            raise NotImplementedError("Void function have not been implemented")
            #self.error(line,col,f"expressão de retorno inválida. Procedimentos não podem retornar valores")
        elif function.type.name != node.exp.eval_type and node.exp.prom_type == None:
            self.error(line,col,f"expressão de retorno inválida. O tipo do valor de retorno é incompatível com o tipo de retorno da função")




    def visit_se(self,node):
        self.visit(node.condition)
        if node.condition.eval_type != Type.BOOL:
            self.error(node.token.line,f"a condição da instrução 'se' deve ser um valor lógico")
        self.visit(node.then_branch)
        if node.else_branch:
            self.visit(node.else_branch)

    def visit_enquanto(self,node):
        self.visit(node.condition)
        if node.condition.eval_type != Type.BOOL:
            self.error(node.token.line,f"a condição da instrução 'enquanto' deve ser um valor lógico")
        self.visit(node.statement)

    def visit_para(self,node):
        self.visit(node.expression)
        name = node.expression.name.lexeme
        sym = SYM.VariableSymbol(name,self.current_scope.resolve("int"))
        scope = SYM.Scope(SYM.Scope.LOCAL,self.current_scope)
        scope.define(name,sym)
        self.visit(node.statement,scope)

    def visit_paraexpr(self,node):
        self.visit(node.name)
        self.visit(node.range_expr)

    def visit_rangeexpr(self,node):
        self.visit(node.start)
        self.visit(node.end)
        if node.inc:
            self.visit(node.inc)
        for node in (node.start,node.end,node.inc):
            #Skip inc node in case it's empty lool
            if not node:
                continue
            if node.eval_type != Type.INT:
                self.error(node.token.line,"os parâmetros de uma série devem ser do tipo 'int'")

    def visit_call(self,node):
        line = node.token.line
        col = node.token.col
        callee = node.callee
        if isinstance(callee,ast.Variable):
            name = callee.token.lexeme
            sym = self.current_scope.resolve(name)
        elif isinstance(calle,ast.Call):
            sym = self.visit(node)
            name = sym.name
        else:
            raise NotImplementedError("Don't know what to do with anything else in call")
        if not sym:
            self.error(line,col,f"o identificador '{callee}' não foi definido neste escopo")
        if not sym.is_callable():
            self.error(line,col,f"identificador '{callee}' não é invocável")
        for arg in node.fargs:
            self.visit(arg)
        arg_len = len(node.fargs)
        param_len = len(sym.params)
        if arg_len != param_len:
            self.error(
                        line,col,
                        f"número incorrecto de argumentos para a função {callee}. Esperava {param_len} argumentos, porém recebeu {arg_len}"
                    )
        #Type promotion for parameter
        for arg,param in zip(node.fargs,sym.params.values()):
            arg.prom_type = type_promotion[arg.eval_type.value][param.type.name.value]
            if param.type.name != arg.eval_type and arg.prom_type == None:
                self.error(line,col,f"argumento inválido. Esperava-se um argumento do tipo '{param.param_type.name}' mas recebeu o tipo '{arg.eval_type}'.")
        node.eval_type = sym.type.name
        return sym


    
        
