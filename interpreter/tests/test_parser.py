import unittest
from interpreter.parser import Parser
from interpreter.lexer import Lexer
from interpreter.tokens import TokenType,Token
from io import StringIO


#buffer = StringIO("")



class ParserTestCase(unittest.TestCase):
    def setUp(self):
        self.buffer = StringIO()

    def tearDown(self):
        self.buffer = None

    def test_declaration(self):
        phrases = [ "decl int a;","declara real a1;","declara bool a2;",
            "declara real a3;", "decl real troco = 3.14;", "vector int array[2];"
            "vector int array[2+1];","vector string array[(-(-2))+2];"
        ]
        self.buffer.writelines(phrases)
        self.buffer.seek(0)
        parser = Parser(Lexer(self.buffer))
        parser.parse()

    def test_expression(self):
        phrases = ["2-1;","2+1;","2/1;","2*1;","2%1;",
            "'string'+'op';","2.132+1;","'string'*2;","string*2;",
            "string[0]*5;","a+b-c*array[1]%(-a)/(-c)+eval(2+1,5);","+--2---5;"
            "'string'/2.241;","(c*array[1]+soma(1-3))/((2.132+1)*('string'*2));",
            "a;","add(1-2);","array[1];","a = b;","a = func(a);",
            "a[0] = a;","b[0] = func(a);","a=b=c=d=a[0];","a=b=c=d=func(b);",""
        ]
        for phrase in phrases:
            print(phrase,file=self.buffer)
        #self.buffer.writelines(phrases)
        self.buffer.seek(0)
        parser = Parser(Lexer(self.buffer))
        parser.parse()



if __name__=="__main__":
    unittest.main()
