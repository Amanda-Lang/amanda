use crate::alloc::{Alloc, Ref};
use crate::errors::AmaErr;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

pub type FuncArgs<'a, 'args> = &'args [Ref<'a>];

#[derive(Clone, Copy)]
pub struct NativeFunc<'a> {
    pub name: &'a str,
    pub func: fn(FuncArgs<'a, '_>, &mut Alloc<'a>) -> Result<Ref<'a>, AmaErr>,
}

impl<'a> Debug for NativeFunc<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "NativeFunc").unwrap();
        Ok(())
    }
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