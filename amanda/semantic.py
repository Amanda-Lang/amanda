from enum import Enum
from amanda.lexer import Lexer
from amanda.tokens import TokenType as TT
from amanda.tokens import Token
import amanda.ast_nodes as ast
import amanda.symbols as SYM
import amanda.error as error
import amanda.natives as natives



'''
Class that performs semantic analysis on a syntatically valid
amanda program.
'''

class Analyzer(ast.Visitor):

    def __init__(self,src):

        #Just to have quick access to things like types and e.t.c
        self.global_scope = SYM.Scope(SYM.Scope.GLOBAL)
        self.current_scope = self.global_scope
        self.src = src 
        self.current_node = None
        self.init_builtins()

    def init_builtins(self):
        self.global_scope.define("int",SYM.BuiltInType("int",SYM.Tag.INT))
        self.global_scope.define("real",SYM.BuiltInType("real",SYM.Tag.REAL))
        self.global_scope.define("bool",SYM.BuiltInType("bool",SYM.Tag.BOOL))
        builtins = natives.builtin_types.values()

        for type_obj in builtins:
            type_obj.load_symbol(self.global_scope)
        
        for type_obj in builtins:
            type_obj.define_symbol(self.global_scope)
        

    def has_return(self,node):
        node_class = type(node).__name__.lower()
        method_name = f"has_return_{node_class}"
        visitor_method = getattr(self,method_name,self.general_check)
        #update AST node
        self.current_node = node
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


    def error(self,code,**kwargs):
        message = code.format(**kwargs)
        handler = error.ErrorHandler.get_handler()
        handler.throw_error(
            error.Analysis(
              message,self.current_node.token.line,
              self.current_node.token.col
              ),
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
        var_type = self.current_scope.resolve(node.var_type.lexeme)
        if not var_type or not var_type.is_type():
            self.error(
                        error.Analysis.UNDEFINED_TYPE,
                        type=node.var_type.lexeme
                    )

        if self.current_scope.resolve(name):
            self.error(error.Analysis.ID_IN_USE,name=name)
        symbol = SYM.VariableSymbol(name,var_type)
        self.current_scope.define(name,symbol)
        assign = node.assign
        if assign is not None:
            if assign.right.token.lexeme == name:
                self.error(f"Erro ao inicializar variável. Não pode referenciar uma variável durante a sua declaração")
            self.visit(assign)


    def visit_functiondecl(self,node):
        #Check if id is already in use
        name = node.name.lexeme
        if self.current_scope.resolve(name):
            self.error(error.Analysis.ID_IN_USE,name=name)
        #Check if return types exists
        decl_type = node.func_type.lexeme
        function_type =  self.current_scope.resolve(decl_type)
        #TODO: fix this hack
        if not function_type or not function_type.is_type():
            self.error(error.Analysis.UNDEFINED_TYPE,type=decl_type)
        has_return = self.has_return(node.block)
        if not has_return:
            #TODO: fix this hack
            self.current_node = node
            self.error(error.Analysis.NO_RETURN_STMT,name=name)
        symbol = SYM.FunctionSymbol(name,function_type)
        self.current_scope.define(name,symbol)
        scope,symbol.params = self.define_func_scope(name,node.params)
        self.visit(node.block,scope)
        

    def define_func_scope(self,name,params):
        params_dict = {}
        for param in params:
            param_name = param.name.lexeme
            if params_dict.get(param_name):
                self.error(error.Analysis.REPEAT_PARAM,name=param_name)
            param_symbol = self.visit(param)
            params_dict[param_name] = param_symbol
            #Add params o current_scope
        scope = SYM.Scope(name,self.current_scope)
        for param_name,param in params_dict.items():
            scope.define(param_name,param)
        return (scope,params_dict)



    def visit_classdecl(self,node):
        name = node.name.lexeme
        if self.current_scope.resolve(name):
            self.error(error.Analysis.ID_IN_USE,name=name)
        #Check if class has a valid superclass
        superclass = node.superclass
        if superclass:
            #check superclass here
            pass
        symbol = SYM.ClassSymbol(name,superclass=superclass)
        self.current_scope.define(name,symbol)
        #Do some superclass stuff here
        #//////////////////////////////
        scope = SYM.Scope(name,self.current_scope)
        members = self.visit_classbody(node.body,scope)
        symbol.members = members

    
    def visit_classbody(self,node,scope):
        self.current_scope = scope
        for child in node.children:
            self.visit(child)
        self.current_scope = self.current_scope.enclosing_scope
        return scope.symbols


    def visit_block(self,node,scope=None):
        if not scope:
            scope = SYM.Scope(SYM.Scope.LOCAL,self.current_scope)
        self.current_scope = scope
        for child in node.children:
            self.visit(child)
        self.current_scope = self.current_scope.enclosing_scope


    def visit_param(self,node):
        param_type = node.param_type.lexeme
        var_type = self.current_scope.resolve(param_type)
        if not var_type or not var_type.is_type():
            self.error(
                        node.param_type.line,
                        node.param_type.col,
                        error.Analysis.UNDEFINED_TYPE,
                        type=param_type
                    )
        name = node.name.lexeme
        return SYM.VariableSymbol(name,var_type)


    def visit_constant(self,node):

        constant = node.token.token
        if constant == TT.INTEGER:
            node.eval_type = self.global_scope.resolve("int")
        elif constant == TT.REAL:
            node.eval_type = self.global_scope.resolve("real")
        elif constant == TT.STRING:
            raise Exception("Not implemented strings yet")
        elif constant in (TT.VERDADEIRO,TT.FALSO):
            node.eval_type = self.global_scope.resolve("bool")


    def visit_variable(self,node):
        name = node.token.lexeme
        sym = self.current_scope.resolve(name)
        if not sym:
            self.error(error.Analysis.UNDECLARED_ID,name=name)
        #Referencing array by name 
        elif not sym.can_evaluate():
            self.error(error.Analysis.INVALID_REF,name=name)
        node.eval_type = sym.type


    def visit_binop(self,node):
        self.visit(node.left)
        self.visit(node.right)
        #Evaluate type of binary
        #arithmetic operation
        operator = node.token
        lhs = node.left
        rhs = node.right
        result = lhs.eval_type.validate_op(operator.token,rhs.eval_type,self.current_scope)
        if not result:
            self.current_node = node
            self.error(
                error.Analysis.INVALID_OP,
                t1=lhs.eval_type,
                t2=rhs.eval_type,
                operator=operator.lexeme)
            
        node.eval_type = result
        lhs.prom_type = lhs.eval_type.promote_to(rhs.eval_type,self.current_scope)
        rhs.prom_type = rhs.eval_type.promote_to(lhs.eval_type,self.current_scope)


    def visit_unaryop(self,node):
        self.visit(node.operand)
        operator = node.token.token
        lexeme = node.token.lexeme
        op_type = node.operand.eval_type
        if operator in (TT.PLUS,TT.MINUS):
            if op_type.tag != SYM.Tag.INT and op_type.tag != Tag.REAL:
                self.current_node = node
                self.error(error.Analysis.INVALID_UOP,operator=lexeme,op_type=op_type)
        elif operator == TT.NAO:
            if op_type.tag != SYM.Tag.BOOL:
                self.error(error.Analysis.INVALID_UOP,operator=lexeme,type=op_type)
        node.eval_type = op_type


    def visit_assign(self,node):
        lhs = node.left
        rhs = node.right
        
        self.visit(rhs)
        self.visit(lhs)

        #Set node types
        node.eval_type = lhs.eval_type
        node.prom_type = None
        #Set promotion type for right side
        rhs.prom_type = rhs.eval_type.promote_to(lhs.eval_type,self.current_scope)
        if lhs.eval_type != rhs.eval_type and not rhs.prom_type:
            self.current_node = node
            self.error(f"atribuição inválida. incompatibilidade entre os operandos da atribuição")


    def visit_mostra(self,node):
        self.visit(node.exp)


    def visit_retorna(self,node):
        expr = node.exp
        self.visit(expr)
        function = self.current_scope.get_enclosing_func()
        if not function:
            self.current_node = node
            self.error(f"O comando 'retorna' só pode ser usado dentro de uma função")
        func_type = function.type
        #TODO: Work out what to do about void types
        if not function.type:
            raise NotImplementedError("Void functions have not been implemented")
        expr.prom_type = expr.eval_type.promote_to(func_type,self.current_scope)
        if func_type.tag != expr.eval_type.tag and not expr.prom_type:
            self.error(f"expressão de retorno inválida. O tipo do valor de retorno é incompatível com o tipo de retorno da função")



    def visit_se(self,node):
        self.visit(node.condition)
        if node.condition.eval_type.tag != SYM.Tag.BOOL:
            self.error(node.token.line,f"a condição da instrução 'se' deve ser um valor lógico")
        self.visit(node.then_branch)
        if node.else_branch:
            self.visit(node.else_branch)

    def visit_enquanto(self,node):
        self.visit(node.condition)
        if node.condition.eval_type.tag != SYM.Tag.BOOL:
            print("LOGGING",node.condition.eval_type)
            self.current_node = node
            self.error(f"a condição da instrução 'enquanto' deve ser um valor lógico")
        self.visit(node.statement)

    def visit_para(self,node):
        self.visit(node.expression)
        #Define control variable for loop
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
            if node.eval_type.tag != SYM.Tag.INT:
                self.error("os parâmetros de uma série devem ser do tipo 'int'")

    def visit_call(self,node):
        callee = node.callee
        #Call is made on a variable
        if isinstance(callee,ast.Variable):
            name = callee.token.lexeme
            sym = self.current_scope.resolve(name)
        #Call is made on another call
        elif isinstance(callee,ast.Call):
            sym = self.visit(callee)
            name = sym.name
            #check if sym return type is callable
            if not sym.type.is_callable():
                self.error(f"Valores do tipo '{sym.type.name}' não são invocáveis")
            return sym
        else:
            raise NotImplementedError("Don't know what to do with anything else in call")
        self.validate_call(sym,node.fargs)
        node.eval_type = sym.type        
        return sym


    def validate_call(self,sym,fargs):
        ''' Helper method that enforces a host of semantic 
        checks on a call operation. '''
        name = sym.name
        if not sym:
            self.error(f"o identificador '{name}' não foi definido neste escopo")
        if not sym.is_callable():
            self.error(f"identificador '{name}' não é invocável")
        for arg in fargs:
            self.visit(arg)
        arg_len = len(fargs)
        param_len = sym.arity()
        if arg_len != param_len:
            self.error(
                f"número incorrecto de argumentos para a função {name}. Esperava {param_len} argumento(s), porém recebeu {arg_len}")
        #Type promotion for parameter
        for arg,param in zip(fargs,sym.params.values()):
            arg.prom_type = arg.eval_type.promote_to(param.type,self.current_scope)
            if param.type.tag != arg.eval_type.tag and not arg.prom_type:
                self.error(
                   f"argumento inválido. Esperava-se um argumento do tipo '{param.param_type.name}' mas recebeu o tipo '{arg.eval_type}'")
        
    
        
