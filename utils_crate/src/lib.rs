// utils_crate/src/lib.rs

use serde::{Deserialize, Serialize};

/// Parameters for controlling the sampling process during text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature for sampling. Higher values make the output more random.
    /// A value of 0.0 or very close to 0.0 will result in greedy decoding (argmax).
    pub temperature: f32,

    /// Top-P (nucleus) sampling probability. Filters the vocabulary to the smallest set of
    /// tokens whose cumulative probability exceeds top_p. 0.0 < top_p <= 1.0.
    /// A value of 1.0 disables Top-P sampling.
    pub top_p: f32,

    /// Top-K sampling. Filters the vocabulary to the K most likely next tokens.
    /// A value of 0 disables Top-K sampling.
    pub top_k: usize,

    /// Seed for the random number generator to ensure reproducibility.
    /// If None, a random seed may be used.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // Default values similar to those in cli_app and api_server_crate
        Self {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 0, // Disabled by default
            seed: None,
        }
    }
}

/// Represents a task for the inference engine, including the prompt
/// and sampling parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTask {
    /// The input prompt string for the language model.
    pub prompt: String,

    /// Parameters to control the generation process.
    pub sampling_params: SamplingParams,
}
