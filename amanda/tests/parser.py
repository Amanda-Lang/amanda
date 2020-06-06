import unittest
from amanda.parser import Parser
from amanda.lexer import Lexer
from amanda.tokens import TokenType,Token
from io import StringIO


#buffer = StringIO("")



class ParserTestCase(unittest.TestCase):
    def setUp(self):
        self.buffer = StringIO()

    def tearDown(self):
        self.buffer = None

    def test_declaration(self):
        phrases = [ "var a: int","var a1: real","var a2: bool",
            "var a3:real", "var troco : real = 3.14", 
        ]
        for phrase in phrases:
            print(phrase,file=self.buffer,end="\n\n\n")
        self.buffer.seek(0)
        parser = Parser(Lexer(self.buffer))
        parser.parse()

    def test_expression(self):
        phrases = ["2-1;","2+1;","2/1;","2*1;","2%1;","2+ad;",
            "'string'+'op';","2.132+1;","'string'*2;","string*2;",
            "string[0]*5;","a+b-c*array[1]%(-a)/(-c)+eval(2+1,5);","+--2---5;"
            "'string'/2.241;","(c*array[1]+soma(1-3))/((2.132+1)*('string'*2));",
            "a;","add(1-2);","array[1];","a = b;","a = func(a);",
            "a[0] = a;","b[0] = func(a);","a=b=c=d=a[0];","a=b=c=d=func(b);","(a+b>a-b);",
            "(((a-b>=a+c)<(a-b))<=(a*2+5)) ou falso != !verdadeiro;","a += 1;a-=2*1;a[0]*=4*(76-2);"
        ]
        for phrase in phrases:
            print(phrase,file=self.buffer)
        #self.buffer.writelines(phrases)
        self.buffer.seek(0)
        parser = Parser(Lexer(self.buffer))
        parser.parse()

    def test_statement(self):
        phrases = ["mostra 2-1;","retorna eval(2*1+((21-1)*12));","retorna a[0];",
            "retorna a[0];","mostra a+b-c*array[1]%(-a)/(-c)+eval(2+1,5);",'''
            se (verdadeiro == falso) entao  rebenta;
            se (1<2)  entao
                rouba();
            se (1==2) entao
                devolve();
            senao
                vai_preso();
            se (0==0) entao {
                rebenta;
            }
            senao{
                fecha;
            }
            se (verdadeiro == falso) entao rebenta();
            senao fecha();

            {
                var int a;
            }

            enquanto(verdadeiro) faca{
                rebenta;
            }
            enquanto(verdadeiro) faca
                rebenta;

            enquanto(verdadeiro) faca{
                mostra a;
            }

            para (i de 0..9) faca
                mostra i;

            para (i de 0..9) faca{
                mostra i+1;
                uno;
                0+2;
            }
            para (i de num..num+3) faca{
                mostra i+1;
                uno;
                0+2;
            }
            para (i de num..num+3 inc num-1) faca{
                mostra i+1;
                uno;
                0+2;
            }
            '''
        ]
        for phrase in phrases:
            print(phrase,file=self.buffer)
        #self.buffer.writelines(phrases)
        self.buffer.seek(0)
        parser = Parser(Lexer(self.buffer))
        parser.parse()

    def test_function_decl(self):
        phrases = ['''
            func test(a:int,b:int): int
                2-1
                var soma : int = a+b
                #mostra a+b-c*array[1]%(-a)/(-c)+eval(2+1,5)
                retorna -soma+(2*2%1)
            fim

            func test(a:int,b:int): int
                2-1;
                var soma : int = a+b;
                #mostra a+b-c*array[1]%(-a)/(-c)+eval(2+1,5);
                retorna -soma+(2*2%1);
            fim
            


            func test(): int 

            fim

            func test(): vazio 

            fim

            func test(a:int,b:int): vazio 

            fim

        '''
        ]
        for phrase in phrases:
            print(phrase,file=self.buffer)
        #self.buffer.writelines(phrases)
        self.buffer.seek(0)
        parser = Parser(Lexer(self.buffer))
        parser.parse()




if __name__=="__main__":
    unittest.main()
