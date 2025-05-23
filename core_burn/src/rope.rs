// core_burn/src/rope.rs

use burn::tensor::{backend::Backend, Tensor, Float, Int, Shape};
use burn::module::Module;
use core::f32::consts::PI; // Using f32::consts::PI

/// Configuration for Rotary Positional Embedding (RoPE).
#[derive(Debug, Clone, Module)]
pub struct RoPEConfig {
    /// The number of dimensions for the rotary embedding per head.
    /// Typically, this is a fraction of the head dimension.
    pub dim: usize,
    /// The base for the sinusoidal frequencies. Default is often 10000.0.
    pub base: f32,
    /// Maximum sequence length for which to precompute embeddings.
    pub max_seq_len: usize,
}

/// Implements Rotary Positional Embedding (RoPE).
///
/// RoPE encodes absolute positional information by rotating pairs of input features
/// based on their position. It has shown strong performance in transformer models.
#[derive(Module, Debug)]
pub struct RotaryPositionalEmbedding<B: Backend> {
    config: RoPEConfig,
    cos_cached: Tensor<B, 2>, // Shape: (max_seq_len, dim / 2)
    sin_cached: Tensor<B, 2>, // Shape: (max_seq_len, dim / 2)
}

impl<B: Backend> RotaryPositionalEmbedding<B> {
    /// Creates a new RoPE module.
    ///
    /// # Arguments
    ///
    /// * `config`: Configuration for RoPE.
    /// * `device`: The device on which to allocate tensors.
    ///
    /// # Panics
    ///
    /// Panics if `config.dim` is not an even number, as RoPE works on pairs of features.
    pub fn new(config: RoPEConfig, device: &B::Device) -> Self {
        if config.dim % 2 != 0 {
            panic!("RoPE dimension (config.dim) must be an even number.");
        }

        // Precompute cosine and sine frequencies
        // Equivalent to:
        // theta = 1.0 / (base^( (0..dim by 2) / dim ))
        // freqs = (0..max_seq_len).outer_product(theta)
        // cos_cached = freqs.cos()
        // sin_cached = freqs.sin()

        let inv_freq: Vec<f32> = (0..config.dim)
            .step_by(2)
            .map(|i| 1.0 / (config.base.powf(i as f32 / config.dim as f32)))
            .collect();
        let inv_freq_tensor = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device).unsqueeze(); // Shape [1, dim/2]

        let t = Tensor::<B, 1>::arange_device(0..config.max_seq_len, device)
            .float()
            .unsqueeze_dim(1); // Shape [max_seq_len, 1]
        
        let freqs = t.matmul(inv_freq_tensor); // Shape [max_seq_len, dim/2]

        let cos_cached = freqs.clone().cos();
        let sin_cached = freqs.sin();
        
        Self {
            config,
            cos_cached,
            sin_cached,
        }
    }

    /// Applies RoPE to the input tensor (query or key).
    ///
    /// # Arguments
    ///
    /// * `x`: Input tensor, typically query or key.
    ///        Expected shape (batch_size, num_heads, seq_len, head_dim).
    /// * `seq_len_offset`: The offset for sequence length, used when caching KV pairs.
    ///                     If `seq_len_offset > 0`, it means we are processing tokens
    ///                     beyond the initial sequence, and RoPE should be applied
    ///                     relative to their actual positions.
    ///
    /// # Returns
    ///
    /// Tensor with RoPE applied. Shape remains the same as input `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x.dims()[2] + seq_len_offset > self.config.max_seq_len`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, seq_len_offset: usize) -> Tensor<B, D> {
        // D is typically 4 for (batch_size, num_heads, seq_len, head_dim)
        // The RoPE is applied to the last dimension (head_dim) up to self.config.dim
        if D < 2 { // Need at least seq_len and feature_dim
            panic!("Input tensor must have at least 2 dimensions for RoPE.");
        }

        let x_dims = x.dims();
        let seq_len_dim_idx = D - 2; // Assuming features are the last dim, seq_len is second to last
        let current_seq_len = x_dims[seq_len_dim_idx];
        
        if current_seq_len + seq_len_offset > self.config.max_seq_len {
            panic!(
                "Sequence length {} with offset {} exceeds RoPE's precomputed max_seq_len {}.",
                current_seq_len, seq_len_offset, self.config.max_seq_len
            );
        }

        // Get the relevant slice of cached sin/cos based on seq_len_offset and current_seq_len
        let cos = self.cos_cached.clone().slice([
            seq_len_offset..(seq_len_offset + current_seq_len),
            0..(self.config.dim / 2),
        ]); // Shape [current_seq_len, dim/2]
        let sin = self.sin_cached.clone().slice([
            seq_len_offset..(seq_len_offset + current_seq_len),
            0..(self.config.dim / 2),
        ]); // Shape [current_seq_len, dim/2]

        // Reshape cos and sin to be broadcastable with x
        // Target shape for cos/sin: (1, 1, current_seq_len, dim/2) for D=4
        // Or more generally, broadcastable to match x's seq_len and feature dimensions.
        let mut cos_sin_shape_vec = vec![1; D];
        cos_sin_shape_vec[seq_len_dim_idx] = current_seq_len;
        cos_sin_shape_vec[D - 1] = self.config.dim / 2; // Apply to half of the rotary dim
        
        let cos_reshaped = cos.reshape(Shape::new(cos_sin_shape_vec.clone()));
        let sin_reshaped = sin.reshape(Shape::new(cos_sin_shape_vec));

        // Split x into two halves for rotation: x1 and x2
        // x_rope contains the part of x to which RoPE is applied (up to self.config.dim)
        // x_pass contains the part of x that is not affected by RoPE
        
        let (x_rope, x_pass) = if x_dims[D-1] > self.config.dim {
            let split_idx = self.config.dim;
            (x.clone().slice_dims(D-1, 0..split_idx), x.clone().slice_dims(D-1, split_idx..x_dims[D-1]))
        } else {
            (x.clone(), Tensor::empty(Shape::from(x_dims).with_dim(D-1, 0), &x.device())) // Empty tensor if all dims are used
        };

        let x1 = x_rope.clone().slice_dims(D-1, 0..(self.config.dim / 2));
        let x2 = x_rope.slice_dims(D-1, (self.config.dim / 2)..self.config.dim);

        // Apply rotations:
        // x_out1 = x1 * cos - x2 * sin
        // x_out2 = x2 * cos + x1 * sin
        let x_out1 = x1.clone() * cos_reshaped.clone() - x2.clone() * sin_reshaped.clone();
        let x_out2 = x2 * cos_reshaped + x1 * sin_reshaped;

        // Concatenate rotated parts
        let rotated_x = Tensor::cat(vec![x_out1, x_out2], D - 1);

        // Concatenate with the pass-through part if it exists
        if x_pass.dims()[D-1] > 0 {
            Tensor::cat(vec![rotated_x, x_pass], D-1)
        } else {
            rotated_x
        }
    }
}
```
