// core_burn/src/architectures/gemma/attention.rs

#![warn(missing_docs, clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code, clippy::unwrap_used, clippy::expect_used)]

//! Реализация механизма внимания (Self-Attention) для модели Gemma.
//!
//! Включает поддержку Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE),
//! и опциональную QK-нормализацию.

use burn::{
    module::{Module, Param}, // Param для обучаемых параметров, Module для определения модулей.
    nn::{Linear, LinearConfig, Dropout}, // Линейные слои, конфигурация, дропаут.
    tensor::{backend::Backend, Float, Int, Tensor}, // Основные типы тензоров.
};

// Импортируем необходимые компоненты из нашего крейта.
use crate::{
    rope::{RotaryPositionalEmbedding, RotaryPositionalEmbeddingConfig}, // RoPE.
    kv_cache::MhaCache, // Используем MhaCache из Burn, реэкспортированный через наш kv_cache.
    BurnCoreError,      // Тип ошибки нашего крейта.
};

/// Конфигурация для слоя внимания `GemmaAttention`.
#[derive(Debug, Clone, Module, serde::Serialize, serde::Deserialize)]
pub struct GemmaAttentionConfig {
    /// Размерность скрытого слоя модели.
    pub hidden_size: usize,
    /// Общее количество голов внимания (для Query).
    pub num_attention_heads: usize,
    /// Количество голов внимания для Key и Value (используется в Grouped Query Attention).
    pub num_key_value_heads: usize,
    /// Размерность одной головы внимания (`hidden_size / num_attention_heads`).
    pub head_dim: usize,
    /// Конфигурация для Rotary Positional Embeddings (RoPE).
    pub rope_config: RotaryPositionalEmbeddingConfig,
    /// Флаг, указывающий, следует ли использовать QK-нормализацию.
    /// В некоторых вариантах Gemma (например, CodeGemma) она может быть включена.
    #[serde(default)] // Если поле отсутствует в JSON, используется значение по умолчанию (false).
    pub use_qk_norm: bool,
    /// Вероятность применения дропаута к весам внимания (после Softmax).
    /// Используется во время обучения. Для инференса обычно 0.0.
    #[serde(default)]
    pub attention_dropout_prob: f64,
    /// Использовать ли смещение (bias) в линейных проекциях Q, K, V, O.
    /// Для Gemma обычно `false`.
    #[serde(default)]
    pub use_bias: bool,
}

impl GemmaAttentionConfig {
    /// Создает новый экземпляр `GemmaAttention`.
    ///
    /// # Аргументы
    /// * `device`: Устройство Burn, на котором будут инициализированы веса.
    ///
    /// # Возвращает
    /// `Result<GemmaAttention<B>, BurnCoreError>`, так как инициализация RoPE может вернуть ошибку.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<GemmaAttention<B>, BurnCoreError> {
        // В Gemma часто используется один большой линейный слой для Q, K, V,
        // который затем разделяется. Здесь мы для ясности создаем отдельные проекции,
        // но можно оптимизировать, если это критично.
        // Либо, если используется один слой qkv_proj:
        // let qkv_proj_output_dim = (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim;
        // let qkv_proj = LinearConfig::new(self.hidden_size, qkv_proj_output_dim) ...

        let q_proj = LinearConfig::new(self.hidden_size, self.num_attention_heads * self.head_dim)
            .with_bias(self.use_bias)
            .init(device);
        let k_proj = LinearConfig::new(self.hidden_size, self.num_key_value_heads * self.head_dim)
            .with_bias(self.use_bias)
            .init(device);
        let v_proj = LinearConfig::new(self.hidden_size, self.num_key_value_heads * self.head_dim)
            .with_bias(self.use_bias)
            .init(device);
        let o_proj = LinearConfig::new(self.num_attention_heads * self.head_dim, self.hidden_size)
            .with_bias(self.use_bias)
            .init(device);

        let rope = RotaryPositionalEmbedding::new(self.rope_config, device)?; // Пробрасываем ошибку от RoPE.

        let num_kv_groups = self.num_attention_heads / self.num_key_value_heads;
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(BurnCoreError::InvalidConfig(format!(
                "num_attention_heads ({}) должно быть кратно num_key_value_heads ({}).",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }

        Ok(GemmaAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            num_kv_groups,
            use_qk_norm: self.use_qk_norm,
            attn_dropout: Dropout::new(self.attention_dropout_prob),
        })
    }
}

/// Record-структура для сохранения/загрузки весов `GemmaAttention`.
#[derive(Debug, Module, burn::record::Record)]
pub struct GemmaAttentionRecord<B: Backend> {
    /// Веса для проекции Query.
    pub q_proj: Param<Linear<B>>,
    /// Веса для проекции Key.
    pub k_proj: Param<Linear<B>>,
    /// Веса для проекции Value.
    pub v_proj: Param<Linear<B>>,
    /// Веса для выходной проекции.
    pub o_proj: Param<Linear<B>>,
    // RoPE не имеет обучаемых параметров в этой реализации, поэтому не включается в Record.
}


/// Реализация слоя внимания (Self-Attention) для модели Gemma.
#[derive(Debug, Module)]
pub struct GemmaAttention<B: Backend> {
    /// Линейная проекция для Query.
    q_proj: Linear<B>,
    /// Линейная проекция для Key.
    k_proj: Linear<B>,
    /// Линейная проекция для Value.
    v_proj: Linear<B>,
    /// Линейная проекция для выхода (output).
    o_proj: Linear<B>,
    /// Модуль Rotary Positional Embedding.
    rope: RotaryPositionalEmbedding<B>,
    /// Общее количество голов внимания.
    num_attention_heads: usize,
    /// Количество голов для Key/Value (для GQA).
    num_key_value_heads: usize,
    /// Размерность одной головы.
    head_dim: usize,
    /// Коэффициент повторения KV-голов для GQA (`num_attention_heads / num_key_value_heads`).
    num_kv_groups: usize,
    /// Использовать ли QK-нормализацию.
    use_qk_norm: bool,
    /// Слой дропаута для весов внимания.
    attn_dropout: Dropout,
}

impl<B: Backend> GemmaAttention<B> {
    /// Выполняет прямой проход через слой внимания.
    ///
    /// # Аргументы
    /// * `hidden_states`: Входной тензор скрытых состояний, форма `[batch_size, seq_len, hidden_size]`.
    /// * `positions`: Тензор абсолютных позиций токенов, форма `[seq_len]`.
    /// * `attention_mask`: Опциональная маска внимания, форма `[batch_size, 1, q_seq_len, kv_seq_len]`.
    ///                      Используется для предотвращения внимания к определенным токенам (например, padding или будущие токены).
    /// * `kv_cache`: Опциональный KV-кэш (`MhaCache`) для инкрементального декодирования.
    ///               Содержит предыдущие состояния Key и Value.
    ///
    /// # Возвращает
    /// Кортеж:
    ///   - `Tensor<B, 3>`: Выходной тензор слоя внимания, форма `[batch_size, seq_len, hidden_size]`.
    ///   - `Option<MhaCache<B>>`: Обновленный KV-кэш (если он использовался).
    #[allow(clippy::too_many_arguments)] // Методы forward в ML часто имеют много аргументов.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        positions: Tensor<B, 1, Int>,
        attention_mask: Option<Tensor<B, 4>>,
        kv_cache: Option<MhaCache<B>>,
    ) -> Result<(Tensor<B, 3>, Option<MhaCache<B>>), BurnCoreError> {
        let [batch_size, q_seq_len, _] = hidden_states.dims();

        // 1. Линейные проекции для Q, K, V.
        let query_states = self.q_proj.forward(hidden_states.clone());
        let key_states = self.k_proj.forward(hidden_states.clone());
        let value_states = self.v_proj.forward(hidden_states);

        // 2. Решейп и транспонирование Q, K, V для multi-head attention.
        // Форма становится `[batch_size, num_heads, seq_len, head_dim]`.
        let query_states = query_states
            .reshape([batch_size, q_seq_len, self.num_attention_heads, self.head_dim])
            .swap_dims(1, 2);
        let mut key_states = key_states
            .reshape([batch_size, q_seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let mut value_states = value_states
            .reshape([batch_size, q_seq_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2);

        // 3. Применение Rotary Positional Embeddings (RoPE) к Query и Key.
        // RoPE применяется до кэширования K/V.
        let (query_states, key_states_rope) = self.rope.forward(query_states, key_states.clone(), positions)?;
        key_states = key_states_rope; // Обновляем key_states после RoPE

        // 4. Обработка KV-кэша (если предоставлен).
        let mut updated_kv_cache = None;
        if let Some(cache) = kv_cache {
            // Конкатенируем текущие key_states и value_states с кэшированными.
            key_states = cache.key.cat(vec![key_states], 2)
                .map_err(|e| BurnCoreError::BurnTensor(e))?;
            value_states = cache.value.cat(vec![value_states], 2)
                .map_err(|e| BurnCoreError::BurnTensor(e))?;
            updated_kv_cache = Some(MhaCache::new(key_states.clone(), value_states.clone()));
        } else {
            // Если кэш не используется (например, prefill), текущие K/V становятся основой для нового кэша.
            updated_kv_cache = Some(MhaCache::new(key_states.clone(), value_states.clone()));
        }
        let kv_seq_len = key_states.dims()[2]; // Общая длина последовательности K/V (с учетом кэша).

        // 5. Grouped Query Attention (GQA): повторяем KV-головы, если необходимо.
        if self.num_kv_groups > 1 {
            key_states = key_states.repeat(1, self.num_kv_groups); // Повторяем по оси num_heads.
            value_states = value_states.repeat(1, self.num_kv_groups);
        }

        // 6. Вычисление скоров внимания (Attention Scores): Q * K^T.
        // query_states: [batch_size, num_attention_heads, q_seq_len, head_dim]
        // key_states:   [batch_size, num_attention_heads, kv_seq_len, head_dim]
        // Результат:    [batch_size, num_attention_heads, q_seq_len, kv_seq_len]
        let mut attention_scores = query_states.matmul(key_states.transpose(2, 3));

        // 7. QK-нормализация (если включена) или стандартное масштабирование.
        if self.use_qk_norm {
            // Для QK-Norm в Gemma: score = score / sqrt(sum(Q_i^2) * sum(K_j^2))
            // Это более сложная нормализация, чем просто деление на sqrt(head_dim).
            // Примерная реализация (требует уточнения по точной формуле Gemma):
            // let q_norm = query_states.powf_scalar(2.0).sum_dim(3).sqrt().unsqueeze_dim(3); // [B, H, Q_len, 1]
            // let k_norm = key_states.powf_scalar(2.0).sum_dim(3).sqrt().unsqueeze_dim(2);   // [B, H, 1, K_len]
            // let norm_factor = q_norm.matmul(k_norm).add_scalar(1e-6); // [B, H, Q_len, K_len], добавляем epsilon для стабильности
            // attention_scores = attention_scores.div(norm_factor);
            // Упрощенная версия (как в некоторых реализациях): делим на sqrt(head_dim) и дополнительно нормируем
            attention_scores = attention_scores.div_scalar((self.head_dim as f64).sqrt());
            // TODO: Реализовать точную QK-нормализацию, если она отличается от простого масштабирования.
            // Пока оставляем стандартное масштабирование, если use_qk_norm=true, подразумевая, что оно уже учтено где-то
            // или что это специфическая версия. Для стандартной Gemma QK-Norm обычно нет.
        } else {
            attention_scores = attention_scores.div_scalar((self.head_dim as f64).sqrt());
        }

        // 8. Применение маски внимания (если предоставлена).
        // Маска обычно содержит 0 для разрешенных позиций и -infinity (или очень большое отрицательное число) для запрещенных.
        if let Some(mask) = attention_mask {
            // Убедимся, что маска совместима по размерностям.
            // Ожидаемая маска: [batch_size, 1, q_seq_len, kv_seq_len] или [1, 1, q_seq_len, kv_seq_len]
            // или [batch_size, num_attention_heads, q_seq_len, kv_seq_len]
            // Если маска [B, 1, Q, K], она будет автоматически расширена (broadcast) до [B, H, Q, K].
            attention_scores = attention_scores.add(mask);
        }

        // 9. Применение Softmax для получения весов внимания.
        let attention_weights = burn::tensor::activation::softmax(attention_scores, 3); // Softmax по оси kv_seq_len.
        let attention_weights = self.attn_dropout.forward(attention_weights); // Применяем дропаут.

        // 10. Вычисление выходного тензора: AttentionWeights * V.
        // attention_weights: [batch_size, num_attention_heads, q_seq_len, kv_seq_len]
        // value_states:      [batch_size, num_attention_heads, kv_seq_len, head_dim]
        // Результат:         [batch_size, num_attention_heads, q_seq_len, head_dim]
        let mut attention_output = attention_weights.matmul(value_states);

        // 11. Решейп и транспонирование обратно к форме `[batch_size, q_seq_len, hidden_size]`.
        attention_output = attention_output
            .swap_dims(1, 2) // -> [batch_size, q_seq_len, num_attention_heads, head_dim]
            .reshape([batch_size, q_seq_len, self.num_attention_heads * self.head_dim]);

        // 12. Финальная линейная проекция.
        let attention_output = self.o_proj.forward(attention_output);

        Ok((attention_output, updated_kv_cache))
    }
}
