// core_burn/src/architectures/gemma.rs

use burn::{
    module::Module,
    tensor::{
        backend::Backend,
        Tensor, Float, Bool, Int,
        ops::{TensorOps, ModuleOps}, 
        Shape, 
    },
    nn::{self, Gelu, Embedding}, 
};
use crate::kv_cache::KvCache; 
use crate::rope::{RotaryPositionalEmbedding, RoPEConfig}; 

// Configuration for the Gemma Model
#[derive(Debug, Clone, Module)]
pub struct GemmaConfig {
    pub hidden_size: usize,
    pub rms_norm_eps: f64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

impl GemmaConfig {
    pub fn new(
        hidden_size: usize,
        rms_norm_eps: f64,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        intermediate_size: usize,
        rope_dim: usize, 
        max_position_embeddings: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            hidden_size,
            rms_norm_eps,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim: hidden_size / num_attention_heads,
            intermediate_size,
            rope_dim, 
            rope_theta: 10000.0,
            max_position_embeddings,
            vocab_size,
        }
    }
}

#[derive(Module, Debug, Clone)]
pub struct RMSNorm<B: Backend> {
    weight: nn::parameters::Param<Tensor<B, 1>>,
    variance_epsilon: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: nn::parameters::Param::from(Tensor::ones([dim], device)),
            variance_epsilon: eps,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let variance = x.clone().mul(x.clone()).mean_dim(D - 1);
        let inv_std = variance.add_scalar(self.variance_epsilon).sqrt().recip();
        let inv_std_broadcastable = inv_std.unsqueeze_dim(D - 1);
        let x_norm = x * inv_std_broadcastable;
        x_norm * self.weight.val().unsqueeze_dims(&[0; D-1][..])
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
    activation: Gelu,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &GemmaConfig, device: &B::Device) -> Self {
        let linear_config = nn::LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false);
        let down_linear_config = nn::LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(false);
        Self {
            gate_proj: nn::Linear::new(linear_config.clone(), device),
            up_proj: nn::Linear::new(linear_config, device),
            down_proj: nn::Linear::new(down_linear_config, device),
            activation: Gelu::new(),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = self.gate_proj.forward(x.clone());
        let gate_activated = self.activation.forward(gate);
        let up = self.up_proj.forward(x);
        let S_gated = gate_activated * up;
        self.down_proj.forward(S_gated)
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    o_proj: nn::Linear<B>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope: RotaryPositionalEmbedding<B>,
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &GemmaConfig, device: &B::Device) -> Self {
        let q_total_dim = config.num_attention_heads * config.head_dim; 
        let kv_total_dim = config.num_key_value_heads * config.head_dim;
        let q_config = nn::LinearConfig::new(config.hidden_size, q_total_dim).with_bias(false);
        let k_config = nn::LinearConfig::new(config.hidden_size, kv_total_dim).with_bias(false);
        let v_config = nn::LinearConfig::new(config.hidden_size, kv_total_dim).with_bias(false);
        let o_config = nn::LinearConfig::new(q_total_dim, config.hidden_size).with_bias(false);
        let rope_config = RoPEConfig {
            dim: config.rope_dim, 
            base: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
        };
        Self {
            q_proj: nn::Linear::new(q_config, device),
            k_proj: nn::Linear::new(k_config, device),
            v_proj: nn::Linear::new(v_config, device),
            o_proj: nn::Linear::new(o_config, device),
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope: RotaryPositionalEmbedding::new(rope_config, device),
        }
    }

    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let num_reps = self.num_attention_heads / self.num_key_value_heads;
        if num_reps == 1 { x } else { x.repeat(1, num_reps) }
    }
    
    pub fn forward( &self, x: Tensor<B, 3>, seq_len_offset: usize, cache: Option<&mut KvCache<B, 4>>, mask: Option<Tensor<B, 4>> ) -> Tensor<B, 3> {
        let (batch_size, q_seq_len, _hidden_size) = x.dims();
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q_reshaped = q.reshape(Shape::new([batch_size, q_seq_len, self.num_attention_heads, self.head_dim])).transpose(1, 2);
        let k_reshaped = k.reshape(Shape::new([batch_size, q_seq_len, self.num_key_value_heads, self.head_dim])).transpose(1, 2);
        let v_reshaped = v.reshape(Shape::new([batch_size, q_seq_len, self.num_key_value_heads, self.head_dim])).transpose(1, 2);
        
        let q_rope = self.rope.forward(q_reshaped, seq_len_offset);
        let k_rope = self.rope.forward(k_reshaped, seq_len_offset);

        let (k_final, v_final) = if let Some(cache_mut) = cache {
            cache_mut.update(k_rope, v_reshaped, 2)
        } else { (k_rope, v_reshaped) };
        
        let k_repeated = self.repeat_kv(k_final); 
        let v_repeated = self.repeat_kv(v_final); 

        let scores = q_rope.matmul(k_repeated.transpose(2, 3)) / (self.head_dim as f64).sqrt();
        // In the previous version of Attention, the mask was `m.equal_elem(0.0f32)`.
        // The problem description's mask logic for GemmaModel's forward is `mask_fill(m, -inf)`.
        // This implies `m` itself is the boolean mask where `true` means "mask this token".
        // Let's assume the mask passed to Attention's forward is already a boolean mask.
        let scores_masked = if let Some(m) = mask { scores.mask_fill(m, -B::FloatElem::INFINITY) } else { scores };
        let attn_weights = nn::softmax(scores_masked, 3); 
        let context = attn_weights.matmul(v_repeated); 
        let context_reshaped = context.transpose(1, 2).reshape(Shape::new([batch_size, q_seq_len, self.num_attention_heads * self.head_dim]));
        self.o_proj.forward(context_reshaped)
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    self_attn: Attention<B>,
    mlp: Mlp<B>,
    input_layernorm: RMSNorm<B>,
    post_attention_layernorm: RMSNorm<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(config: &GemmaConfig, device: &B::Device) -> Self {
        Self {
            self_attn: Attention::new(config, device),
            mlp: Mlp::new(config, device),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps, device),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps, device),
        }
    }

    pub fn forward( &self, x: Tensor<B, 3>, seq_len_offset: usize, cache: Option<&mut KvCache<B, 4>>, mask: Option<Tensor<B, 4>>, ) -> Tensor<B, 3> {
        let normalized_x = self.input_layernorm.forward(x.clone());
        let attn_output = self.self_attn.forward(normalized_x, seq_len_offset, cache, mask);
        let x_after_attn = x + attn_output; 
        let normalized_x_after_attn = self.post_attention_layernorm.forward(x_after_attn.clone());
        let mlp_output = self.mlp.forward(normalized_x_after_attn);
        x_after_attn + mlp_output
    }
}

// Full Gemma Model
#[derive(Module, Debug)]
pub struct GemmaModel<B: Backend> {
    embed_tokens: nn::Embedding<B>,
    blocks: Vec<DecoderBlock<B>>, // For Burn, if training, this would ideally be a ModuleList or similar
                                  // to ensure parameters are discovered. For inference, direct Vec is fine.
    norm: RMSNorm<B>,
    config: GemmaConfig, // Store config for direct use (e.g. scaling factor)
}

impl<B: Backend> GemmaModel<B> {
    pub fn new(config: GemmaConfig, device: &B::Device) -> Self {
        let embed_config = nn::EmbeddingConfig::new(config.vocab_size, config.hidden_size);
        let embed_tokens = nn::Embedding::new(embed_config, device);

        // Create DecoderBlocks
        // Note: For proper parameter registration in Burn if training,
        // one would typically use a more structured way like a ModuleList,
        // or ensure each block is a direct field for Module derive to pick up.
        // Since this seems inference-focused, direct Vec storage is simpler.
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for _i in 0..config.num_hidden_layers { // _i is conventional for unused loop var
            blocks.push(DecoderBlock::new(&config, device));
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps, device);
        
        Self {
            embed_tokens,
            blocks,
            norm,
            config, // Clones the config
        }
    }

    /// Main forward pass for the GemmaModel.
    ///
    /// # Arguments
    /// * `input_ids`: Tensor of input token IDs. Shape: (batch_size, seq_len).
    /// * `seq_len_offset`: Offset for sequence length, used for KV caching and RoPE.
    ///                     This indicates the number of tokens already processed and cached.
    /// * `kv_caches`: A mutable vector of `KvCache`s, one for each decoder layer.
    ///                If `None`, KV caching is disabled.
    /// * `mask`: Optional attention mask. Assumed to be a boolean mask where `true` indicates
    ///           a position that should be masked (e.g., filled with -infinity in scores).
    ///
    /// # Returns
    /// Tensor of last hidden states. Shape: (batch_size, seq_len, hidden_size).
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        seq_len_offset: usize,
        kv_caches: Option<&mut Vec<KvCache<B, 4>>>, // One cache per layer
        mask: Option<Tensor<B, 4>>, // Boolean mask, true = mask out
    ) -> Tensor<B, 3, Float> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Gemma-specific scaling of embeddings: hidden_states * sqrt(hidden_size)
        let scaling_factor = (self.config.hidden_size as f64).sqrt();
        hidden_states = hidden_states * scaling_factor;

        if let Some(caches_mut_vec) = kv_caches {
            // Ensure correct number of caches if provided
            if caches_mut_vec.len() != self.config.num_hidden_layers {
                panic!(
                    "Number of KV caches ({}) does not match number of hidden layers ({}).",
                    caches_mut_vec.len(), self.config.num_hidden_layers
                );
            }
            for (i, block) in self.blocks.iter().enumerate() {
                // Get mutable reference to the i-th cache
                let cache_opt = Some(&mut caches_mut_vec[i]);
                hidden_states = block.forward(hidden_states, seq_len_offset, cache_opt, mask.clone());
            }
        } else {
            // No KV caching
            for block in self.blocks.iter() {
                hidden_states = block.forward(hidden_states, seq_len_offset, None, mask.clone());
            }
        }
        
        self.norm.forward(hidden_states)
        // To return logits for generation, a final linear layer (lm_head) would be needed here:
        // e.g. self.lm_head.forward(final_hidden_states)
        // where lm_head would be typically nn::Linear::new(config.hidden_size, config.vocab_size)
        // and might share weights with embed_tokens in some models.
    }
}
```
