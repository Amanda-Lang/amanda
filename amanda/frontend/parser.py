from amanda.frontend.lexer import Lexer
from amanda.frontend.tokens import TokenType as TT
from amanda.frontend.tokens import Token
import amanda.frontend.ast as ast
from amanda.error import AmandaError


class Parser:
    #Errors messages
    MISSING_TERM = "as instruções devem ser delimitadas por ';' ou por uma nova linha"
    ILLEGAL_EXPRESSION = "início inválido de expressão"
    EXPECTED_ID = "era esperado um identificador depois do símbolo '{symbol}'"
    EXPECTED_TYPE = "era esperado um tipo depois do símbolo '{symbol}'"
    ILLEGAL_ASSIGN = "alvo inválido para atribuição"

    def __init__(self,io_object):
        self.lexer = Lexer(io_object)
        self.lookahead = self.lexer.get_token()

    def consume(
        self,expected,
        error=None,skip_newlines=False
    ):
        if skip_newlines:
            self.skip_newlines()
        if self.match(expected):
            consumed = self.lookahead
            self.lookahead = self.lexer.get_token()
            return consumed
        else:
            if error:
                self.error(error)
            self.error(f"era esperado o símbolo {expected.value},porém recebeu o símbolo '{self.lookahead.lexeme}'")

    def error(self,message,line=None):
        err_line = self.lookahead.line
        if line:
            err_line = line
        raise AmandaError.syntax_error(
             message,err_line,
             self.lookahead.col
         )

    def skip_newlines(self):
        while self.match(TT.NEWLINE):
            self.lookahead = self.lexer.get_token()

    def parse(self):
        return self.program()

    def match(self,token):
        return self.lookahead.token == token

    def program(self):
        program = ast.Program() 
        while not self.match(Lexer.EOF):
            if self.match(TT.NEWLINE):
                self.consume(TT.NEWLINE)
            else:
                child = self.declaration()
                self.append_child(program,child)
        return program

    def block(self):
        block = ast.Block()
        #SENAO is because of if statements
        while not self.match(TT.FIM) and not self.match(TT.SENAO):
            if self.match(TT.NEWLINE):
                self.consume(TT.NEWLINE)
            else:
                child = self.declaration()
                self.append_child(block,child)
        return block

    def append_child(self,body,child):
        ''' Method for desugaring
        multiple statement'''
        if isinstance(child,list):
            body.children += child
        else:
            body.add_child(child)

    def declaration(self):
        if self.match(TT.FUNC):
            return self.function_decl()
        elif self.match(TT.CLASSE):
            return self.class_decl()
        else:
            return self.statement()

    def type(self):
        is_list = False
        if self.match(TT.LBRACKET):
            self.consume(TT.LBRACKET)
            self.consume(TT.RBRACKET)
            is_list = True
        type_name = self.consume(TT.IDENTIFIER)
        return ast.Type(type_name,is_list)

    def end_stmt(self):
        if self.match(TT.NEWLINE):
            self.consume(TT.NEWLINE)
        elif self.match(TT.SEMI):
            self.consume(TT.SEMI)
        else:
            self.error(self.MISSING_TERM)

    def function_decl(self):
        self.consume(TT.FUNC)
        name = self.consume(
            TT.IDENTIFIER,self.EXPECTED_ID.format(symbol="func")
        )
        self.consume(TT.LPAR)
        params = self.formal_params()
        self.consume(TT.RPAR,"os parâmetros da função devem estar delimitados por  ')'")
        func_type = None
        if self.match(TT.COLON):
            self.consume(TT.COLON)
            if self.match(TT.VAZIO): #REMOVE: Maybe remove this
                self.consume(TT.VAZIO)
            else:
                func_type = self.type()
        block = self.block()
        self.consume(TT.FIM,"O corpo de um função deve ser terminado com a directiva 'fim'")
        return ast.FunctionDecl(name=name,block=block,func_type=func_type,params=params)

    def class_decl(self):
        self.consume(TT.CLASSE)
        name = self.consume(TT.IDENTIFIER)
        body = self.class_body()
        self.consume(TT.FIM,"O corpo de uma classe deve ser terminado com o símbolo fim")
        return ast.ClassDecl(name=name,body=body)

    def class_body(self):
        body = ast.ClassBody()
        while not self.match(TT.FIM):
            if self.match(TT.NEWLINE):
                self.skip_newlines()
            else:
                member = self.declaration()
                member_type = type(member)
                if member_type != ast.FunctionDecl and \
                member_type != ast.VarDecl:
                    self.error(
                        "directiva inválida para o corpo de uma classe",
                        member.lineno
                    )
                if member_type == ast.VarDecl and \
                member.assign is not None:
                    #TODO: Fix line number error here
                    self.error(
                        "Não pode inicializar os campos de um classe",
                        member.lineno
                    )
                body.add_child(member)
        return body

    def formal_params(self):
        params = []
        self.skip_newlines()
        if self.lookahead.token == TT.IDENTIFIER:
            name = self.consume(TT.IDENTIFIER,skip_newlines=True)
            self.consume(TT.COLON,"esperava-se o símbolo ':'.")
            param_type = self.type()
            params.append(ast.Param(param_type,name))
            while not self.match(TT.RPAR):
                self.consume(TT.COMMA,skip_newlines=True)
                name = self.consume(TT.IDENTIFIER,skip_newlines=True)
                self.consume(TT.COLON,"esperava-se o símbolo ':'.")
                param_type = self.type()
                params.append(ast.Param(param_type,name))
                self.skip_newlines()
        return params

    def statement(self):
        if self.match(TT.MOSTRA):
            return self.mostra_statement()
        elif self.match(TT.RETORNA):
            return self.retorna_statement()
        elif self.match(TT.ENQUANTO):
            return self.enquanto_stmt()
        elif self.match(TT.SE):
            return self.se_statement()
        elif self.match(TT.PARA):
            return self.para_stmt()
        else:
            return self.decl_stmt()

    def mostra_statement(self):
        token = self.consume(TT.MOSTRA)
        exp = self.equality()
        self.end_stmt()
        return ast.Mostra(token,exp)

    def retorna_statement(self):
        token = self.consume(TT.RETORNA)
        exp = self.equality()
        self.end_stmt()
        return ast.Retorna(token,exp)

    def se_statement(self):
        token = self.consume(TT.SE)
        condition = self.equality()
        self.consume(TT.ENTAO)
        then_branch = self.block()
        else_branch = None
        if self.match(TT.SENAO):
            self.consume(TT.SENAO)
            else_branch = self.block()
        self.consume(TT.FIM,"esperava-se a símbolo fim para terminar a directiva 'se'")
        return ast.Se(token,condition,then_branch,else_branch)

    def enquanto_stmt(self):
        token = self.consume(TT.ENQUANTO)
        condition = self.equality()
        self.consume(TT.FACA)
        block = self.block()
        self.consume(TT.FIM,"esperava-se o símbolo fim para terminar a directiva 'enquanto'")
        return ast.Enquanto(token,condition,block)

    def para_stmt(self):
        token = self.consume(TT.PARA)
        expression = self.for_expression()
        self.consume(TT.FACA)
        block = self.block()
        self.consume(TT.FIM,"esperava-se o símbolo fim para terminar a directiva 'para'")
        return ast.Para(token,expression,block)

    def for_expression(self):
        name = self.consume(TT.IDENTIFIER)
        self.consume(TT.DE)
        range_expr = self.range_expression(name)
        return ast.ParaExpr(name,range_expr)

    def range_expression(self,token):
        start = self.equality()
        self.consume(TT.DDOT)
        stop = self.equality()
        inc = None
        if self.lookahead.token == TT.INC:
            self.consume(TT.INC)
            inc = self.equality()
        return ast.RangeExpr(token,start,stop,inc)

    def decl_stmt(self):
        stmt = self.expression()
        if isinstance(stmt,ast.Variable):
            if self.match(TT.COLON):
                stmt = self.simple_decl(stmt.token)
            elif self.match(TT.COMMA):
                stmt = self.multi_decl(stmt.token)
        self.end_stmt()
        return stmt

    def get_decl_assign(self,name):
        assign = None
        if self.match(TT.EQUAL):
            assign = ast.Assign(
                self.consume(TT.EQUAL),
                left=ast.Variable(name),
                right = self.equality()
            )
        return assign


    def simple_decl(self,name):
        token = self.consume(TT.COLON)
        var_type = self.type()
        assign = self.get_decl_assign(name)
        return ast.VarDecl(
                token,name=name,var_type=var_type,
                assign=assign
        )

    def multi_decl(self,name):
        names = []
        names.append(name)
        while self.match(TT.COMMA):
            self.consume(TT.COMMA)
            name = self.consume(
                TT.IDENTIFIER,self.EXPECTED_ID.format(symbol=",")
            )
            names.append(name)
        token = self.consume(TT.COLON)
        var_type = self.type()
        decls = []
        for var_name in names:
            decl = ast.VarDecl(
                token,name=var_name,var_type=var_type,
                assign=None
            )
            decls.append(decl)
        return decls

    def expression(self):
        return self.compound_assignment()

    def eq_operator(self):
        if self.match(TT.DOUBLEEQUAL):
            return self.consume(TT.DOUBLEEQUAL)
        elif self.match(TT.NOTEQUAL):
            return self.consume(TT.NOTEQUAL)

    def compound_assignment(self):
        expr = self.assignment()
        compound_operator = (TT.PLUSEQ,TT.MINUSEQ,TT.SLASHEQ,TT.STAREQ)
        current = self.lookahead.token
        if current in compound_operator:
            if not expr.is_assignable():
                self.error(self.ILLEGAL_ASSIGN)
            #Create separate tokens
            token = Token(None,None,line=self.lookahead.line,col=self.lookahead.col)
            eq = Token(TT.EQUAL,"=",line=self.lookahead.line,col=self.lookahead.col)
            token.token,token.lexeme = self.compound_operator()
            self.consume(current)
            if isinstance(expr,ast.Get):
                expr = ast.Set(target=expr,expr=self.assignment()) 
            else:
                expr = ast.Assign(eq,left=expr,right=ast.BinOp(token,left=expr,right=self.equality()))
        return expr

    def compound_operator(self):
        if self.match(TT.PLUSEQ):
            op = (TT.PLUS,"+")
        elif self.match(TT.MINUSEQ):
            op = (TT.MINUS,"-")
        elif self.match(TT.STAREQ):
            op = (TT.STAR,"*")
        elif self.match(TT.SLASHEQ):
            op = (TT.SLASH,"/")
        return op
 
    def assignment(self):
        expr = self.equality()
        if self.match(TT.EQUAL):
            token = self.consume(TT.EQUAL)
            if not expr.is_assignable():
                self.error(self.ILLEGAL_ASSIGN)
            if isinstance(expr,ast.Get):
                expr = ast.Set(target=expr,expr=self.assignment()) 
            else:
                expr = ast.Assign(token,left=expr,right=self.assignment())
        return expr

    def equality(self):
        node = self.comparison()
        while self.lookahead.token in (TT.DOUBLEEQUAL,TT.NOTEQUAL):
            op = self.eq_operator()
            node = ast.BinOp(op,left=node,right=self.comparison())
        return node

    def comp_operator(self):
        if self.match(TT.GREATER):
            return self.consume(TT.GREATER)
        elif self.match(TT.GREATEREQ):
            return self.consume(TT.GREATEREQ)
        elif self.match(TT.LESS):
            return self.consume(TT.LESS)
        elif self.match(TT.LESSEQ):
            return self.consume(TT.LESSEQ)

    def comparison(self):
        node = self.addition()
        while self.lookahead.token in (TT.GREATER,TT.GREATEREQ,TT.LESS,TT.LESSEQ):
            op = self.comp_operator()
            node = ast.BinOp(op,left=node,right=self.addition())
        return node

    def addition(self):
        node = self.term()
        while self.lookahead.token in (TT.PLUS,TT.MINUS,TT.OU):
            if self.match(TT.OU):
                op = self.consume(TT.OU)
            else:
                op = self.add_operator()
            node = ast.BinOp(op,left=node,right=self.term())
        return node

    def term(self):
        node = self.unary()
        while self.lookahead.token in (
            TT.STAR,TT.DOUBLESLASH,
            TT.SLASH,TT.MODULO,TT.E
        ):
            op = self.mult_operator()
            node = ast.BinOp(op,left=node,right=self.unary())
        return node

    def unary(self):
        current = self.lookahead.token
        if current in (TT.PLUS,TT.MINUS,TT.NAO):
            token = self.consume(current)
            expr = ast.UnaryOp(token,operand=self.unary())
            return expr
        return self.call()
        
    def call(self):
        expr = self.primary()
        while self.lookahead.token in\
        (TT.LPAR,TT.DOT,TT.LBRACKET):
            if self.match(TT.LPAR):
                self.consume(TT.LPAR)
                args = []
                args = self.args()
                token = self.consume(
                    TT.RPAR,
                    "os argumentos da função devem ser delimitados por ')'"
                )
                expr = ast.Call(callee=expr,paren=token,fargs=args)
            elif self.match(TT.LBRACKET):
                self.consume(TT.LBRACKET)
                index = self.equality()
                token = self.consume(TT.RBRACKET)
                expr = ast.Index(token,expr,index)
            else:
                self.consume(TT.DOT)
                identifier = self.lookahead
                self.consume(TT.IDENTIFIER)
                expr = ast.Get(target=expr,member=identifier)
        return expr

    def primary(self):
        current = self.lookahead.token
        expr = None
        if current in (TT.INTEGER,TT.REAL,TT.STRING,TT.IDENTIFIER,TT.VERDADEIRO,TT.FALSO):
            if self.match(TT.IDENTIFIER):
                expr = ast.Variable(self.lookahead)
            else:
                expr = ast.Constant(self.lookahead)
            self.consume(current)
        elif self.match(TT.LPAR):
            self.consume(TT.LPAR)
            expr = self.equality()
            self.consume(TT.RPAR)
        elif self.match(TT.EU):
            expr = ast.Eu(self.lookahead)
            self.consume(TT.EU)
        elif self.match(TT.CONVERTE):
            expr = self.converte_expression()
        else:
            self.error(f"início inválido de expressão: '{self.lookahead.lexeme}'")
        return expr
    
    def converte_expression(self):
        token = self.consume(TT.CONVERTE)
        self.consume(TT.LPAR)
        expression = self.equality()
        self.consume(TT.COMMA)
        new_type = self.type()
        self.consume(TT.RPAR)
        return ast.Converte(token,expression,new_type)

    def args(self):
        current = self.lookahead.token
        args = []
        self.skip_newlines()
        if not self.match(TT.RPAR):
            args.append(self.equality())
        while not self.match(TT.RPAR):
            self.consume(TT.COMMA,skip_newlines=True)
            self.skip_newlines()
            args.append(self.equality())
            self.skip_newlines()
        return args

    def mult_operator(self):
        if self.match(TT.E):
            return self.consume(TT.E)
        elif self.match(TT.STAR):
            return self.consume(TT.STAR)
        elif self.match(TT.SLASH):
            return self.consume(TT.SLASH)
        elif self.match(TT.DOUBLESLASH):
            return self.consume(TT.DOUBLESLASH)
        elif self.match(TT.MODULO):
            return self.consume(TT.MODULO)

    def add_operator(self):
        if self.match(TT.PLUS):
            return self.consume(TT.PLUS)
        elif self.match(TT.MINUS):
            return self.consume(TT.MINUS)
