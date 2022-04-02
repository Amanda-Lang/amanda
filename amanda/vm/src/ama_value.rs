use crate::binload::Const;
use crate::vm::OpCode;
use std::borrow::Cow;
use std::convert::From;
use std::fmt;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy)]
enum BinOpResult {
    Int,
    Double,
    Bool,
    Str,
}

macro_rules! arith_ops {
    ($res_type: ident, $left: ident, $op: tt, $right: ident) => {
        Ok(match $res_type {
            BinOpResult::Double => AmaValue::F64($left.take_float() $op $right.take_float()),
            BinOpResult::Int => AmaValue::Int($left.take_int() $op $right.take_int()),
            _ => unimplemented!("Operand type not supported"),
        })
    };
}

macro_rules! comp_ops {
    ($res_type: ident, $left: ident, $op: tt, $right: ident) => {
        Ok(match $res_type {
            BinOpResult::Double => AmaValue::Bool($left.take_float() $op $right.take_float()),
            BinOpResult::Int => AmaValue::Bool($left.take_int() $op $right.take_int()),
            _ => unimplemented!("Operand type not supported"),
        })
    };
}

macro_rules! eq_ops {
    ($res_type: ident, $left: ident, $op: tt, $right: ident) => {
        Ok(match $res_type {
            BinOpResult::Int => AmaValue::Bool($left.take_int() $op $right.take_int()),
            BinOpResult::Double => AmaValue::Bool($left.take_float() $op $right.take_float()),
            BinOpResult::Bool => AmaValue::Bool($left.take_bool() $op $right.take_bool()),
            BinOpResult::Str => AmaValue::Bool($left.take_str() $op $right.take_str()),
        })
    };
}

//TODO: Find a 'safe' way to do this
pub type FuncArgs<'a> = Option<*const AmaValue<'a>>;

#[derive(Debug, Clone, Copy)]
pub struct NativeFunc<'a> {
    pub name: &'a str,
    pub func: fn(FuncArgs<'a>) -> Result<AmaValue<'a>, &'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct AmaFunc<'a> {
    pub name: &'a str,
    pub start_ip: usize,
    pub ip: usize,
    pub last_i: usize,
    pub bp: isize,
    pub locals: usize,
}

/*Primitive Types*/
#[derive(Debug, Copy, Clone)]
pub enum Type {
    Int,
    Real,
    Texto,
    Bool,
}

#[derive(Debug)]
pub enum AmaValue<'a> {
    Str(Cow<'a, String>),
    Int(i64),
    F64(f64),
    Bool(bool),
    Func(AmaFunc<'a>),
    NativeFn(NativeFunc<'a>),
    Type(Type),
    None,
}

impl<'a> AmaValue<'a> {
    pub fn get_type(&self) -> Type {
        match self {
            AmaValue::Str(string) => Type::Texto,
            AmaValue::Int(int) => Type::Int,
            AmaValue::F64(float) => Type::Real,
            AmaValue::Bool(val) => Type::Bool,
            _ => unimplemented!("Cannot return type for this value: {:?}", self),
        }
    }

    pub fn is_float(&self) -> bool {
        if let AmaValue::F64(_) = self {
            true
        } else {
            false
        }
    }

    pub fn take_float(&self) -> f64 {
        match self {
            AmaValue::F64(float) => *float,
            AmaValue::Int(int) => *int as f64,
            _ => panic!("Value is not a float"),
        }
    }

    pub fn take_int(&self) -> i64 {
        match self {
            AmaValue::Int(int) => *int,
            AmaValue::F64(float) => *float as i64,
            _ => panic!("Value is not an int"),
        }
    }

    pub fn take_type(&self) -> Type {
        match self {
            AmaValue::Type(t) => *t,
            _ => panic!("Value is not an int"),
        }
    }

    pub fn take_str(&self) -> &str {
        if let AmaValue::Str(string) = self {
            string
        } else {
            panic!("Value is not an str")
        }
    }

    pub fn take_bool(&self) -> bool {
        if let AmaValue::Bool(val) = self {
            *val
        } else {
            panic!("Value is not a bool")
        }
    }

    pub fn take_func(self) -> AmaFunc<'a> {
        if let AmaValue::Func(func) = self {
            func
        } else {
            panic!("Value is not a func")
        }
    }

    pub fn is_str(&self) -> bool {
        if let AmaValue::Str(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_bool(&self) -> bool {
        if let AmaValue::Bool(_) = self {
            true
        } else {
            false
        }
    }

    pub fn is_int(&self) -> bool {
        if let AmaValue::Int(_) = self {
            true
        } else {
            false
        }
    }

    pub fn binop(left: Self, op: OpCode, right: Self) -> Result<Self, &'a str> {
        let res_type = if left.is_float() || right.is_float() {
            BinOpResult::Double
        } else if left.is_int() && right.is_int() {
            BinOpResult::Int
        } else if left.is_bool() && right.is_bool() {
            BinOpResult::Bool
        } else if left.is_str() && right.is_str() {
            BinOpResult::Str
        } else {
            unimplemented!("Error is not implemented")
        };
        match op {
            OpCode::OpAdd => arith_ops!(res_type, left, +, right),
            OpCode::OpMinus => arith_ops!(res_type, left, -, right),
            OpCode::OpMul => arith_ops!(res_type, left, *, right),
            OpCode::OpModulo => arith_ops!(res_type, left, %, right),
            OpCode::OpDiv => {
                if right.take_float() == 0.0 {
                    Err("não pode dividir um número por zero")
                } else {
                    Ok(AmaValue::F64(left.take_float() / right.take_float()))
                }
            }
            OpCode::OpFloorDiv => {
                if right.take_int() == 0 {
                    Err("não pode dividir um número por zero")
                } else {
                    Ok(AmaValue::Int(left.take_int() / right.take_int()))
                }
            }
            OpCode::OpAnd => Ok(AmaValue::Bool(left.take_bool() && right.take_bool())),
            OpCode::OpOr => Ok(AmaValue::Bool(left.take_bool() || right.take_bool())),
            OpCode::OpEq => eq_ops!(res_type, left, ==, right),
            OpCode::OpNotEq => eq_ops!(res_type, left, !=,  right),
            OpCode::OpGreater => comp_ops!(res_type, left, >, right),
            OpCode::OpGreaterEq => comp_ops!(res_type, left, >=, right),
            OpCode::OpLess => comp_ops!(res_type, left, <, right),
            OpCode::OpLessEq => comp_ops!(res_type, left, <=, right),
            _ => unimplemented!("Op {:?} has not yet been implemented", op),
        }
    }
}

impl<'a> From<&'a Const> for AmaValue<'a> {
    fn from(constant: &Const) -> AmaValue {
        match constant {
            Const::Str(string) => AmaValue::Str(Cow::Borrowed(string)),
            Const::Int(int) => AmaValue::Int(*int),
            Const::Double(real) => AmaValue::F64(*real),
            Const::Bool(boolean) => AmaValue::Bool(*boolean),
        }
    }
}

impl Clone for AmaValue<'_> {
    fn clone(&self) -> Self {
        match self {
            AmaValue::Str(string) => AmaValue::Str(Clone::clone(string)),
            AmaValue::Int(int) => AmaValue::Int(*int),
            AmaValue::F64(float) => AmaValue::F64(*float),
            AmaValue::Bool(val) => AmaValue::Bool(*val),
            AmaValue::None => AmaValue::None,
            AmaValue::Func(function) => AmaValue::Func(*function),
            AmaValue::NativeFn(func) => AmaValue::NativeFn(*func),
            AmaValue::Type(t) => AmaValue::Type(*t),
        }
    }
}

impl Display for AmaValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AmaValue::Str(string) => write!(f, "{}", string),
            AmaValue::Int(integer) => write!(f, "{}", integer),
            AmaValue::F64(float) => {
                if float.fract() == 0.0 {
                    return write!(f, "{}.0", float);
                }
                write!(f, "{}", float)
            }
            AmaValue::Bool(val) => {
                let val_str = if *val { "verdadeiro" } else { "falso" };
                write!(f, "{}", val_str)
            }
            AmaValue::None => panic!("None value should not be printed"),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub fn check_cast(val_t: Type, target: Type) -> bool {
    val_t as u8 == target as u8
}

pub fn cast(value: AmaValue, target: Type) -> Result<AmaValue, &'static str> {
    match target {
        Type::Texto => Ok(AmaValue::Str(Cow::Owned(format!("{}", value)))),
        Type::Int => match value {
            AmaValue::F64(_) => Ok(AmaValue::Int(value.take_int())),
            AmaValue::Str(string) => {
                let maybe_int = string.parse::<i64>();
                if let Err(_) = maybe_int {
                    Err("A sequência de caracteres 'xxx' não é um inteiro válido")
                } else {
                    Ok(AmaValue::Int(maybe_int.unwrap()))
                }
            }
            _ => unimplemented!("Impossível valor para o tipo especificado"),
        },
        Type::Real => match value {
            AmaValue::Int(_) => Ok(AmaValue::F64(value.take_float())),
            AmaValue::Str(string) => {
                let maybe_real = string.parse::<f64>();
                if let Err(_) = maybe_real {
                    Err("A sequência de caracteres 'xxx' não é um número real válido")
                } else {
                    Ok(AmaValue::F64(maybe_real.unwrap()))
                }
            }
            _ => unimplemented!("Impossível valor para o tipo especificado"),
        },
        Type::Bool => match value {
            AmaValue::Int(int) => Ok(AmaValue::Bool(int != 0)),
            AmaValue::F64(double) => Ok(AmaValue::Bool(double != 0.0)),
            AmaValue::Str(string) => Ok(AmaValue::Bool(string.as_ref() != "")),
            _ => unimplemented!("Impossível valor para o tipo especificado"),
        },
    }
}