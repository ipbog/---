// core_burn/src/architectures/gemma/ffn.rs

#![warn(missing_docs, clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code, clippy::unwrap_used, clippy::expect_used)]

//! Реализация полносвязной сети (Feed-Forward Network, FFN) для модели Gemma.
//!
//! Gemma использует вариант FFN, известный как Gated Linear Unit (GLU)
//! с активацией GELU (Gaussian Error Linear Unit).

use burn::{
    module::{Module, Param}, // Module для определения слоя, Param для весов.
    nn::{Linear, LinearConfig, GELU, Dropout}, // Линейные слои, активация GELU, дропаут.
    tensor::{backend::Backend, Tensor}, // Основной тип тензора.
};

/// Конфигурация для слоя `GemmaFeedForward`.
#[derive(Debug, Clone, Module, serde::Serialize, serde::Deserialize)]
pub struct GemmaFeedForwardConfig {
    /// Размерность скрытого слоя модели (входная и выходная размерность FFN).
    pub hidden_size: usize,
    /// Размерность промежуточного слоя в FFN (обычно в 2-4 раза больше `hidden_size`).
    pub intermediate_size: usize,
    /// Вероятность применения дропаута к выходу FFN (после `down_proj`).
    /// Используется во время обучения. Для инференса обычно 0.0.
    #[serde(default)]
    pub ffn_dropout_prob: f64,
    /// Использовать ли смещение (bias) в линейных слоях FFN.
    /// Для Gemma обычно `false`.
    #[serde(default)]
    pub use_bias: bool,
}

impl GemmaFeedForwardConfig {
    /// Создает новый экземпляр `GemmaFeedForward`.
    ///
    /// # Аргументы
    /// * `device`: Устройство Burn, на котором будут инициализированы веса.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GemmaFeedForward<B> {
        // Слой FFN в Gemma состоит из трех линейных проекций:
        // 1. gate_proj: проецирует вход в intermediate_size.
        // 2. up_proj: также проецирует вход в intermediate_size.
        // 3. down_proj: проецирует результат (после активации и умножения) обратно в hidden_size.
        let gate_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(self.use_bias)
            .init(device);
        let up_proj = LinearConfig::new(self.hidden_size, self.intermediate_size)
            .with_bias(self.use_bias)
            .init(device);
        let down_proj = LinearConfig::new(self.intermediate_size, self.hidden_size)
            .with_bias(self.use_bias)
            .init(device);

        // Gemma использует активацию GELU.
        let activation = GELU::new(); // Стандартная реализация GELU в Burn.

        GemmaFeedForward {
            gate_proj,
            up_proj,
            down_proj,
            activation,
            ffn_dropout: Dropout::new(self.ffn_dropout_prob),
        }
    }
}

/// Record-структура для сохранения/загрузки весов `GemmaFeedForward`.
#[derive(Debug, Module, burn::record::Record)]
pub struct GemmaFeedForwardRecord<B: Backend> {
    /// Веса для `gate_proj`.
    pub gate_proj: Param<Linear<B>>,
    /// Веса для `up_proj`.
    pub up_proj: Param<Linear<B>>,
    /// Веса для `down_proj`.
    pub down_proj: Param<Linear<B>>,
}


/// Реализация слоя Feed-Forward Network (FFN) для модели Gemma.
///
/// Этот слой обычно следует за слоем внимания в каждом блоке декодера трансформера.
/// Он состоит из двух линейных проекций для расширения размерности, функции активации,
/// и одной линейной проекции для сжатия размерности обратно.
/// Gemma использует Gated FFN: `output = down_proj(activation(gate_proj(x)) * up_proj(x))`.
#[derive(Debug, Module)]
pub struct GemmaFeedForward<B: Backend> {
    /// Линейная проекция "gate".
    gate_proj: Linear<B>,
    /// Линейная проекция "up".
    up_proj: Linear<B>,
    /// Линейная проекция "down" (выходная).
    down_proj: Linear<B>,
    /// Функция активации (GELU для Gemma).
    activation: GELU,
    /// Слой дропаута.
    ffn_dropout: Dropout,
}

impl<B: Backend> GemmaFeedForward<B> {
    /// Выполняет прямой проход через слой FFN.
    ///
    /// # Аргументы
    /// * `hidden_states`: Входной тензор, обычно формы `[batch_size, seq_len, hidden_size]`.
    ///
    /// # Возвращает
    /// Выходной тензор той же формы, что и входной.
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1. Применяем gate_proj и up_proj к входным данным.
        let gate_output = self.gate_proj.forward(hidden_states.clone());
        let up_output = self.up_proj.forward(hidden_states);

        // 2. Применяем функцию активации (GELU) к выходу up_proj.
        // В некоторых реализациях активация применяется к gate_output или к обоим.
        // Для Gemma (и многих GLU вариантов): activation(up_output) или activation(gate_output).
        // Классический SwiGLU/GeGLU: gate_output * activation(up_output)
        // Будем следовать: gate_output * activation(up_output) (или наоборот, если активация на gate)
        // В оригинальной Gemma: gelu(gate_proj(x)) * up_proj(x)
        let activated_gate = self.activation.forward(gate_output);

        // 3. Поэлементное умножение результатов.
        let intermediate_states = activated_gate.mul(up_output);

        // 4. Применяем down_proj.
        let output_states = self.down_proj.forward(intermediate_states);

        // 5. Применяем дропаут.
        self.ffn_dropout.forward(output_states)
    }
}
