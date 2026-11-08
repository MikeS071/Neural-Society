//! Blackboard: a preallocated, append-only arena of typed slots.
//!
//! Numbers are stored as little-endian digit vectors (least significant first) in one
//! shared `Vec<u8>`; a slot is `(start, len, sign, kind, status)`. Only the most recently
//! allocated slot is writable, so producing a result never moves older data. `reset()`
//! keeps capacity, so steady-state evaluation does no heap allocation.

use crate::specialists::Op;

pub type SlotId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Operand,
    /// Partial products, magnitudes during sign handling, etc.
    Intermediate,
    /// Output of one operator in the plan.
    StepResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pending,
    Done,
}

#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub kind: Kind,
    pub status: Status,
    pub neg: bool,
    pub start: u32,
    pub len: u32,
}

/// A specialist reply whose confidence fell below the router's threshold.
#[derive(Clone, Copy, Debug)]
pub struct LowConf {
    pub call: u64,
    pub op: Op,
    pub args: [u8; 4],
    pub result: u16,
    pub conf: f32,
    /// Replaced by the exact fallback.
    pub escalated: bool,
}

pub struct Blackboard {
    digits: Vec<u8>,
    slots: Vec<Slot>,
    pub low_conf: Vec<LowConf>,
}

impl Blackboard {
    pub fn with_capacity(digits: usize, slots: usize) -> Self {
        Blackboard {
            digits: Vec::with_capacity(digits),
            slots: Vec::with_capacity(slots),
            low_conf: Vec::with_capacity(64),
        }
    }

    pub fn reset(&mut self) {
        self.digits.clear();
        self.slots.clear();
        self.low_conf.clear();
    }

    pub fn alloc(&mut self, kind: Kind) -> SlotId {
        let id = self.slots.len() as SlotId;
        self.slots.push(Slot {
            kind,
            status: Status::Pending,
            neg: false,
            start: self.digits.len() as u32,
            len: 0,
        });
        id
    }

    #[inline]
    pub fn push(&mut self, id: SlotId, d: u8) {
        debug_assert!(d < 10);
        debug_assert_eq!(
            id as usize,
            self.slots.len() - 1,
            "only the newest slot is writable"
        );
        self.digits.push(d);
        self.slots[id as usize].len += 1;
    }

    /// Strip leading zeros, set the sign (zero is never negative) and mark Done.
    pub fn finish(&mut self, id: SlotId, neg: bool) {
        let s = &mut self.slots[id as usize];
        while s.len > 1 && self.digits[(s.start + s.len - 1) as usize] == 0 {
            s.len -= 1;
        }
        if s.len == 0 {
            self.digits.push(0);
            s.len = 1;
        }
        self.digits.truncate((s.start + s.len) as usize);
        s.neg = neg && !(s.len == 1 && self.digits[s.start as usize] == 0);
        s.status = Status::Done;
    }

    pub fn set_kind(&mut self, id: SlotId, kind: Kind) {
        self.slots[id as usize].kind = kind;
    }

    pub fn slot(&self, id: SlotId) -> &Slot {
        &self.slots[id as usize]
    }
    #[inline]
    pub fn len(&self, id: SlotId) -> usize {
        self.slots[id as usize].len as usize
    }
    #[inline]
    pub fn neg(&self, id: SlotId) -> bool {
        self.slots[id as usize].neg
    }
    /// Digit `i` (0 = least significant); 0 past the end.
    #[inline]
    pub fn digit(&self, id: SlotId, i: usize) -> u8 {
        let s = &self.slots[id as usize];
        if i < s.len as usize {
            self.digits[s.start as usize + i]
        } else {
            0
        }
    }
    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }
    pub fn arena_bytes(&self) -> usize {
        self.digits.len()
    }

    pub fn write_decimal(&self, id: SlotId, out: &mut String) {
        if self.neg(id) {
            out.push('-');
        }
        for i in (0..self.len(id)).rev() {
            out.push((b'0' + self.digit(id, i)) as char);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_and_normalization() {
        let mut bb = Blackboard::with_capacity(16, 4);
        let a = bb.alloc(Kind::Operand);
        for d in [3, 2, 1, 0, 0] {
            bb.push(a, d);
        }
        bb.finish(a, true);
        let z = bb.alloc(Kind::StepResult);
        bb.push(z, 0);
        bb.push(z, 0);
        bb.finish(z, true);
        let mut s = String::new();
        bb.write_decimal(a, &mut s);
        s.push(' ');
        bb.write_decimal(z, &mut s);
        assert_eq!(s, "-123 0");
        assert_eq!(bb.arena_bytes(), 4);
        assert_eq!(bb.slot(z).status, Status::Done);
    }
}
