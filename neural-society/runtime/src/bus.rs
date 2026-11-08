//! Message bus. Each specialist runs on its own thread and pulls fixed-size
//! messages from its inbox (`std::sync::mpsc`). Replies go back to the sender's
//! (client's) reply channel. A direct-call backend with the same interface
//! measures pure orchestration overhead without threads or channels.

use crate::specialists::{Op, Specialist};
use std::sync::mpsc::{channel, Receiver, RecvError, Sender, TryRecvError};
use std::thread::JoinHandle;

/// Fixed-size, `Copy` message: 20 bytes, no heap.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Message {
    pub id: u32,
    /// Client (router) id the reply goes to.
    pub from: u16,
    /// Specialist id.
    pub to: u16,
    pub op: u8,
    pub args: [u8; 4],
    pub result: u16,
    pub conf: f32,
}

/// Anything a router can ask a digit-level question.
pub trait Backend {
    fn call(&mut self, op: Op, args: [u8; 4]) -> (u16, f32);
    /// Request/reply pairs so far.
    fn messages(&self) -> u64;
}

/// No threads, no channels: a plain function call into the specialist.
pub struct Direct {
    specs: Vec<Box<dyn Specialist>>,
    msgs: u64,
}

impl Direct {
    pub fn new(specs: Vec<Box<dyn Specialist>>) -> Self {
        assert_specs(&specs);
        Direct { specs, msgs: 0 }
    }
}

impl Backend for Direct {
    #[inline]
    fn call(&mut self, op: Op, args: [u8; 4]) -> (u16, f32) {
        self.msgs += 1;
        self.specs[op as usize].forward(args)
    }
    fn messages(&self) -> u64 {
        self.msgs
    }
}

fn assert_specs(specs: &[Box<dyn Specialist>]) {
    for (i, s) in specs.iter().enumerate() {
        assert_eq!(
            s.op() as usize,
            i,
            "specialist {i} is {:?}; expected one per op in Op order",
            s.op()
        );
    }
}

/// Receive, optionally busy-polling first. Blocking `recv` parks the thread, and every
/// message then costs a futex wake on each side; spinning trades CPU for latency.
/// Spinning yields periodically and falls back to blocking, so an oversubscribed
/// machine degrades instead of livelocking.
#[inline]
fn recv(rx: &Receiver<Message>, spin: bool) -> Result<Message, RecvError> {
    if spin {
        for i in 0..SPIN_ITERS {
            match rx.try_recv() {
                Ok(m) => return Ok(m),
                Err(TryRecvError::Disconnected) => return Err(RecvError),
                Err(TryRecvError::Empty) => {
                    if i % 256 == 255 {
                        std::thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }
    rx.recv()
}

const SPIN_ITERS: u32 = 1 << 16;

/// Threaded bus: one thread + inbox per specialist, one reply channel per client.
pub struct Bus {
    inboxes: Vec<Sender<Message>>,
    replies: Vec<Option<Receiver<Message>>>,
    threads: Vec<JoinHandle<u64>>,
    spin: bool,
}

impl Bus {
    /// `spin`: busy-poll before blocking (see [`recv`]), in specialists and clients.
    pub fn spawn(specs: Vec<Box<dyn Specialist>>, n_clients: usize, spin: bool) -> Bus {
        assert_specs(&specs);
        let (reply_tx, replies): (Vec<_>, Vec<_>) =
            (0..n_clients).map(|_| channel::<Message>()).unzip();
        let mut inboxes = Vec::new();
        let mut threads = Vec::new();
        for mut spec in specs {
            let (tx, rx) = channel::<Message>();
            inboxes.push(tx);
            let reply_tx = reply_tx.clone();
            let name = format!("spec-{}", spec.op().name());
            threads.push(
                std::thread::Builder::new()
                    .name(name)
                    .spawn(move || {
                        let mut handled = 0u64;
                        while let Ok(mut m) = recv(&rx, spin) {
                            let (r, c) = spec.forward(m.args);
                            m.result = r;
                            m.conf = c;
                            handled += 1;
                            if reply_tx[m.from as usize].send(m).is_err() {
                                // client gone; keep serving others
                            }
                        }
                        handled
                    })
                    .expect("spawn specialist thread"),
            );
        }
        Bus {
            inboxes,
            replies: replies.into_iter().map(Some).collect(),
            threads,
            spin,
        }
    }

    /// Take the handle for client `i` (each can be taken once, and moved to its own thread).
    pub fn client(&mut self, i: usize) -> BusClient {
        BusClient {
            id: i as u16,
            inboxes: self.inboxes.clone(),
            rx: self.replies[i].take().expect("client already taken"),
            next_id: 0,
            spin: self.spin,
        }
    }

    /// Close all inboxes and join the specialist threads. Returns messages handled per specialist.
    /// All `BusClient`s must be dropped first, or this blocks.
    pub fn shutdown(self) -> Vec<u64> {
        drop(self.inboxes);
        self.threads
            .into_iter()
            .map(|t| t.join().expect("specialist panicked"))
            .collect()
    }
}

pub struct BusClient {
    id: u16,
    inboxes: Vec<Sender<Message>>,
    rx: Receiver<Message>,
    next_id: u32,
    spin: bool,
}

impl Backend for BusClient {
    fn call(&mut self, op: Op, args: [u8; 4]) -> (u16, f32) {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let m = Message {
            id,
            from: self.id,
            to: op as u16,
            op: op as u8,
            args,
            result: 0,
            conf: 0.0,
        };
        self.inboxes[m.to as usize]
            .send(m)
            .expect("specialist thread died");
        let r = recv(&self.rx, self.spin).expect("specialist thread died");
        debug_assert_eq!(r.id, id);
        (r.result, r.conf)
    }
    fn messages(&self) -> u64 {
        self.next_id as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialists::Exact;

    fn exact() -> Vec<Box<dyn Specialist>> {
        Op::ALL
            .into_iter()
            .map(|o| Box::new(Exact(o)) as Box<dyn Specialist>)
            .collect()
    }

    #[test]
    fn message_is_small() {
        assert!(std::mem::size_of::<Message>() <= 20);
    }

    #[test]
    fn bus_matches_direct() {
        for spin in [false, true] {
            bus_matches_direct_with(spin);
        }
    }

    fn bus_matches_direct_with(spin: bool) {
        let mut bus = Bus::spawn(exact(), 2, spin);
        let (mut c0, mut c1) = (bus.client(0), bus.client(1));
        let mut d = Direct::new(exact());
        for op in Op::ALL {
            for a in op.all_inputs() {
                assert_eq!(c0.call(op, a), d.call(op, a));
                assert_eq!(c1.call(op, a), d.call(op, a));
            }
        }
        assert_eq!(c0.messages() + c1.messages(), d.messages());
        drop((c0, c1));
        let handled: u64 = bus.shutdown().iter().sum();
        assert_eq!(handled, d.messages());
    }
}
