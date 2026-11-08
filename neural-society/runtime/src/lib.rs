//! Neural Society runtime: tiny single-skill specialists, a blackboard, a message bus
//! and an exact-code router that composes them into big-number arithmetic.
//! std only; no Python at runtime.
pub mod baseline;
pub mod blackboard;
pub mod bus;
pub mod model;
pub mod router;
pub mod specialists;
