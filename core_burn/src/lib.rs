// core_burn/src/lib.rs

// Define modules for different components of the core model logic
pub mod architectures;
pub mod kv_cache;
pub mod rope;

// Re-export important items for easier use by other crates
pub use kv_cache::KvCache;
pub use rope::{RotaryPositionalEmbedding, RoPEConfig};
pub use architectures::gemma::{GemmaModel, GemmaConfig}; // Assuming these will be the primary exports
// Potentially re-export other components like Mlp, Attention, DecoderBlock if they need to be accessed directly
// or if helper functions for building the model are exposed.
