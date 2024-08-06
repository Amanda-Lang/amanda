use std::convert::From;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    Mostra,
    LoadConst,
    LoadName,
    OpAdd,
    OpMinus,
    OpMul,
    OpDiv,
    OpFloorDiv,
    OpModulo,
    OpInvert,
    OpAnd,
    OpOr,
    OpNot,
    OpEq,
    OpNotEq,
    OpGreater,
    OpGreaterEq,
    OpLess,
    OpLessEq,
    OpIndexGet,
    OpIndexSet,
    GetGlobal,
    SetGlobal,
    Jump,
    JumpIfFalse,
    GetLocal,
    SetLocal,
    CallFunction,
    Return,
    Cast,
    BuildStr,
    BuildVec,
    LoadRegisto,
    BuildObj,
    GetProp,
    SetProp,
    Unwrap,
    IsNull,
    LoadModuleDef,
    BuildVariant,
    BindMatchArgs,
    MatchVariant,
    Halt = 255,
}

impl From<&u8> for OpCode {
    fn from(number: &u8) -> Self {
        let ops: [OpCode; 42] = [
            OpCode::Mostra,
            OpCode::LoadConst,
            OpCode::LoadName,
            OpCode::OpAdd,
            OpCode::OpMinus,
            OpCode::OpMul,
            OpCode::OpDiv,
            OpCode::OpFloorDiv,
            OpCode::OpModulo,
            OpCode::OpInvert,
            OpCode::OpAnd,
            OpCode::OpOr,
            OpCode::OpNot,
            OpCode::OpEq,
            OpCode::OpNotEq,
            OpCode::OpGreater,
            OpCode::OpGreaterEq,
            OpCode::OpLess,
            OpCode::OpLessEq,
            OpCode::OpIndexGet,
            OpCode::OpIndexSet,
            OpCode::GetGlobal,
            OpCode::SetGlobal,
            OpCode::Jump,
            OpCode::JumpIfFalse,
            OpCode::GetLocal,
            OpCode::SetLocal,
            OpCode::CallFunction,
            OpCode::Return,
            OpCode::Cast,
            OpCode::BuildStr,
            OpCode::BuildVec,
            OpCode::LoadRegisto,
            OpCode::BuildObj,
            OpCode::GetProp,
            OpCode::SetProp,
            OpCode::Unwrap,
            OpCode::IsNull,
            OpCode::LoadModuleDef,
            OpCode::BuildVariant,
            OpCode::BindMatchArgs,
            OpCode::MatchVariant,
        ];
        if *number == 0xff {
            OpCode::Halt
        } else {
            ops[*number as usize]
        }
    }
}
