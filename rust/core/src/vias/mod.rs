//! Moving vias in the optimizer: the chain ends they carry and the fields
//! they make (`moves`), what they see of the members and of each other
//! (`sight`), their free distances obstacle by obstacle (`free`), and the
//! pairwise term of ground vias (`pairs`).

pub mod free;
pub mod moves;
pub mod pairs;
pub mod sight;
