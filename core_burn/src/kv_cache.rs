// core_burn/src/kv_cache.rs

use burn::tensor::{backend::Backend, Tensor, Int, Float};

/// A generic Key-Value (KV) cache for attention mechanisms.
///
/// This structure holds the keys and values for past tokens, allowing
/// the model to reuse these computations during auto-regressive generation,
/// significantly speeding up the process.
///
/// # Type Parameters
///
/// * `B`: The backend to use for tensors (e.g., NdArray, Wgpu, Tch).
/// * `D`: The dimension of the tensors (typically 4 for batch_size x num_heads x seq_len x head_dim).
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend, const D: usize> {
    /// Cached keys. Shape: (batch_size, num_heads, max_seq_len, head_dim)
    keys: Tensor<B, D>,
    /// Cached values. Shape: (batch_size, num_heads, max_seq_len, head_dim)
    values: Tensor<B, D>,
    /// Current sequence length of the cached tensors.
    /// This indicates how many tokens are currently stored in the cache.
    current_seq_len: usize,
    /// The maximum sequence length this cache can hold.
    max_seq_len: usize,
}

impl<B: Backend, const D: usize> KvCache<B, D> {
    /// Creates a new KV cache.
    ///
    /// # Arguments
    ///
    /// * `batch_size`: The number of sequences in a batch.
    /// * `num_heads`: The number of attention heads.
    /// * `max_seq_len`: The maximum sequence length the cache can accommodate.
    /// * `head_dim`: The dimensionality of each attention head.
    /// * `device`: The device on which to allocate the cache tensors.
    ///
    /// # Returns
    ///
    /// A new `KvCache` instance.
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        // Ensure D is appropriate for typical KV cache shapes (e.g., 4D)
        // This is a compile-time check if D is used in tensor shape directly,
        // but we'll construct shapes dynamically here for clarity.
        // The shape is (batch_size, num_heads, max_seq_len, head_dim)
        let shape = [batch_size, num_heads, max_seq_len, head_dim]; 
        // For a generic D, we might need more complex shape construction or assertions.
        // Assuming D=4 for this typical use case.
        if D != 4 {
            // This is a runtime panic, ideally, this constraint would be more compile-time.
            // Using typenum or const generics for specific dimensions could enforce this better.
            panic!("KvCache currently assumes D=4 for shape [batch_size, num_heads, max_seq_len, head_dim]");
        }

        Self {
            keys: Tensor::zeros(shape, device),
            values: Tensor::zeros(shape, device),
            current_seq_len: 0,
            max_seq_len,
        }
    }

    /// Updates the cache with new keys and values.
    ///
    /// # Arguments
    ///
    /// * `new_keys`: The new key tensor to append. Expected shape (batch_size, num_heads, new_seq_len, head_dim).
    /// * `new_values`: The new value tensor to append. Expected shape (batch_size, num_heads, new_seq_len, head_dim).
    /// * `seq_dim`: The dimension index representing the sequence length (typically 2 for 4D tensors).
    ///
    /// # Returns
    ///
    /// A tuple containing the updated full sequence of keys and values from the cache.
    ///
    /// # Panics
    ///
    /// Panics if the new sequence length exceeds the cache's `max_seq_len`.
    pub fn update(
        &mut self,
        new_keys: Tensor<B, D>,
        new_values: Tensor<B, D>,
        seq_dim: usize, // Typically 2 for (B, H, S, D')
    ) -> (Tensor<B, D>, Tensor<B, D>) {
        let new_len = new_keys.dims()[seq_dim];
        if self.current_seq_len + new_len > self.max_seq_len {
            panic!(
                "KV Cache update would exceed max_seq_len. Current: {}, New: {}, Max: {}",
                self.current_seq_len, new_len, self.max_seq_len
            );
        }

        // Define the ranges for slicing and updating
        // Example for D=4 and seq_dim=2:
        // self.keys = self.keys.slice_assign([0..batch, 0..num_heads, start_idx..end_idx, 0..head_dim], new_keys)
        // This is tricky with generic D and seq_dim without more complex range construction.
        // Burn's API for slice_assign requires specific ranges for each dimension.

        // Assuming D=4 and seq_dim=2 for simplicity in this example.
        // A more robust implementation would handle generic D and seq_dim.
        if D == 4 && seq_dim == 2 {
            let [b, h, _, d_h] = self.keys.dims(); // Get other dimensions
            let start_idx = self.current_seq_len;
            let end_idx = self.current_seq_len + new_len;

            // Create ranges for slice_assign
            let ranges_keys = [0..b, 0..h, start_idx..end_idx, 0..d_h];
            let ranges_values = [0..b, 0..h, start_idx..end_idx, 0..d_h];
            
            self.keys = self.keys.clone().slice_assign(ranges_keys, new_keys);
            self.values = self.values.clone().slice_assign(ranges_values, new_values);
        } else {
            // Fallback or panic for unsupported D/seq_dim combinations
            // This part would need a more generic way to construct ranges for slice_assign
            // or a different update strategy if Burn's API doesn't directly support it easily.
            panic!("KV Cache update for D={} and seq_dim={} is not generically implemented yet.", D, seq_dim);
            // As a placeholder, one might just overwrite or error out.
            // For a real implementation, this needs careful handling of tensor manipulations.
        }

        self.current_seq_len += new_len;

        // Return the relevant slice of the cache up to the current sequence length
        self.get_current_sequence(seq_dim)
    }

    /// Retrieves the current sequence of keys and values from the cache.
    ///
    /// # Arguments
    ///
    /// * `seq_dim`: The dimension index representing the sequence length.
    ///
    /// # Returns
    ///
    /// A tuple `(keys, values)` containing tensors sliced up to `current_seq_len`.
    pub fn get_current_sequence(&self, seq_dim: usize) -> (Tensor<B, D>, Tensor<B, D>) {
        if D == 4 && seq_dim == 2 {
            let [b, h, _, d_h] = self.keys.dims();
            let current_keys = self.keys.clone().slice([0..b, 0..h, 0..self.current_seq_len, 0..d_h]);
            let current_values = self.values.clone().slice([0..b, 0..h, 0..self.current_seq_len, 0..d_h]);
            (current_keys, current_values)
        } else {
            panic!("KV Cache get_current_sequence for D={} and seq_dim={} is not generically implemented yet.", D, seq_dim);
        }
    }
    
    /// Clears the cache by resetting the current sequence length.
    pub fn clear(&mut self) {
        self.current_seq_len = 0;
        // Tensors are not zeroed out for efficiency, new updates will overwrite.
    }

    /// Returns the current sequence length stored in the cache.
    pub fn current_length(&self) -> usize {
        self.current_seq_len
    }

    /// Returns the maximum sequence length the cache can hold.
    pub fn max_length(&self) -> usize {
        self.max_seq_len
    }
}
```
