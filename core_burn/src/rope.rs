// core_burn/src/rope.rs

#![warn(missing_docs, clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code, clippy::unwrap_used, clippy::expect_used)]

use burn::{
    module::Module, // Для #[derive(Module)]
    tensor::{backend::Backend, Float, Int, Tensor}, // Основные типы тензоров
};
// Импортируем наш кастомный тип ошибки для возврата в случае проблем с конфигурацией.
use crate::BurnCoreError;

/// Конфигурация для Rotary Positional Embedding (RoPE).
///
/// Содержит параметры, необходимые для инициализации и работы RoPE.
#[derive(Debug, Clone, Copy, Module, serde::Serialize, serde::Deserialize)]
pub struct RotaryPositionalEmbeddingConfig {
    /// Размерность, к которой применяется RoPE. Обычно это `head_dim` (размерность одной головы внимания).
    pub dim: usize,
    /// Базовое значение `theta` (часто 10000.0), используемое для вычисления частот вращения.
    pub base_theta: f64,
    /// Максимальная длина последовательности, для которой будут предвычислены значения `cos` и `sin`.
    pub max_seq_len: usize,
}

impl RotaryPositionalEmbeddingConfig {
    /// Создает новую конфигурацию RoPE.
    ///
    /// # Аргументы
    /// * `dim`: Размерность применения RoPE.
    /// * `base_theta`: Базовое значение `theta`.
    /// * `max_seq_len`: Максимальная длина последовательности.
    pub fn new(dim: usize, base_theta: f64, max_seq_len: usize) -> Self {
        Self { dim, base_theta, max_seq_len }
    }
}

/// Реализация Rotary Positional Embedding (RoPE).
///
/// RoPE применяет вращение к векторам Query и Key в механизме внимания
/// для кодирования относительной позиционной информации.
/// Значения `cos` и `sin` для вращения предвычисляются при инициализации.
#[derive(Debug, Module)] // Module, чтобы можно было включать в другие модули Burn.
pub struct RotaryPositionalEmbedding<B: Backend> {
    /// Сохраненная конфигурация.
    config: RotaryPositionalEmbeddingConfig,
    /// Предвычисленные значения косинусов: `[max_seq_len, dim / 2]`.
    cos_cached: Tensor<B, 2>,
    /// Предвычисленные значения синусов: `[max_seq_len, dim / 2]`.
    sin_cached: Tensor<B, 2>,
}

impl<B: Backend> RotaryPositionalEmbedding<B> {
    /// Создает новый экземпляр `RotaryPositionalEmbedding` с предвычисленными значениями `cos` и `sin`.
    ///
    /// # Аргументы
    /// * `config`: Конфигурация RoPE.
    /// * `device`: Устройство Burn, на котором будут созданы тензоры.
    ///
    /// # Возвращает
    /// `Result<Self, BurnCoreError>`, так как конфигурация может быть невалидной (например, `dim` нечетное).
    pub fn new(config: RotaryPositionalEmbeddingConfig, device: &B::Device) -> Result<Self, BurnCoreError> {
        // Размерность `dim` должна быть четной, так как RoPE работает с парами элементов.
        if config.dim % 2 != 0 {
            return Err(BurnCoreError::InvalidConfig(format!(
                "Размерность для RoPE (dim={}) должна быть четной.",
                config.dim
            )));
        }

        // Вычисляем обратные частоты: inv_freq = 1.0 / (base_theta ^ (2k / dim))
        // где k - это индекс пары от 0 до dim/2 - 1.
        // Используем log-exp трюк для вычисления base_theta ^ (power): exp(power * ln(base_theta)).
        let inv_freq: Tensor<B, 1> = Tensor::arange_step(0..config.dim, 2, device) // 0, 2, 4, ..., dim-2
            .float() // Преобразуем в float
            .div_scalar(config.dim as f64) // (2k / dim)
            .mul_scalar(config.base_theta.ln()) // (2k / dim) * ln(base_theta)
            .exp() // base_theta ^ (2k / dim)
            .recip(); // 1.0 / (base_theta ^ (2k / dim)) -> форма [dim / 2]

        // Создаем тензор позиций от 0 до max_seq_len - 1.
        let t = Tensor::arange(0..config.max_seq_len, device).float(); // Форма [max_seq_len]

        // Вычисляем аргументы для cos/sin: freqs = t * inv_freq (внешнее произведение)
        // t: [max_seq_len, 1], inv_freq: [1, dim / 2] -> freqs: [max_seq_len, dim / 2]
        let freqs = t.unsqueeze_dim(1).matmul(inv_freq.unsqueeze_dim(0));

        // Предвычисляем и сохраняем значения cos и sin.
        let cos_cached = freqs.clone().cos();
        let sin_cached = freqs.sin();

        Ok(Self { config, cos_cached, sin_cached })
    }

    /// Применяет RoPE к входным тензорам Query (`q_states`) и Key (`k_states`).
    ///
    /// # Аргументы
    /// * `q_states`: Тензор Query, обычно формы `[batch_size, num_q_heads, seq_len, head_dim]`.
    /// * `k_states`: Тензор Key, обычно формы `[batch_size, num_kv_heads, seq_len, head_dim]`.
    /// * `positions`: Тензор абсолютных позиций токенов в последовательности, форма `[seq_len]`.
    ///                Используется для выбора нужных предвычисленных `cos`/`sin` значений.
    ///
    /// # Возвращает
    /// Кортеж `(Tensor<B, 4>, Tensor<B, 4>)` с преобразованными Q и K тензорами.
    /// Возвращает `BurnCoreError` в случае ошибки.
    pub fn forward(
        &self,
        q_states: Tensor<B, 4>,
        k_states: Tensor<B, 4>,
        positions: Tensor<B, 1, Int>, // Позиции для текущего фрагмента последовательности
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>), BurnCoreError> {
        // Извлекаем предвычисленные cos/sin для текущих позиций.
        // `select` позволяет выбрать строки из `cos_cached` по индексам из `positions`.
        let cos = self.cos_cached.clone().select(0, positions.clone()); // Форма [seq_len, dim / 2]
        let sin = self.sin_cached.clone().select(0, positions);       // Форма [seq_len, dim / 2]

        // Решейпим cos/sin для бродкастинга к Q и K.
        // Q/K имеют форму [batch, num_heads, seq_len, head_dim].
        // Cos/Sin должны быть совместимы с последними двумя размерностями [seq_len, head_dim/2].
        // Добавляем фиктивные размерности для batch и num_heads.
        let cos = cos.unsqueeze_dim(0).unsqueeze_dim(0); // Форма [1, 1, seq_len, dim / 2]
        let sin = sin.unsqueeze_dim(0).unsqueeze_dim(0); // Форма [1, 1, seq_len, dim / 2]

        // Применяем вращение к Q и K.
        let q_embed = self.apply_rotary_emb_to_tensor(q_states, &cos, &sin)?;
        let k_embed = self.apply_rotary_emb_to_tensor(k_states, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    /// Вспомогательная функция для применения вращения к одному тензору (Q или K).
    ///
    /// # Аргументы
    /// * `x`: Входной тензор (Q или K), форма `[batch, num_heads, seq_len, head_dim]`.
    /// * `cos_val`: Тензор косинусов, форма `[1, 1, seq_len, head_dim/2]`.
    /// * `sin_val`: Тензор синусов, форма `[1, 1, seq_len, head_dim/2]`.
    ///
    /// # Возвращает
    /// Преобразованный тензор той же формы, что и `x`.
    fn apply_rotary_emb_to_tensor(
        &self,
        x: Tensor<B, 4>,
        cos_val: &Tensor<B, 4>,
        sin_val: &Tensor<B, 4>,
    ) -> Result<Tensor<B, 4>, BurnCoreError> {
        let dim_half = self.config.dim / 2;

        // Разделяем тензор `x` на две половины по последней размерности (`head_dim`).
        // x = [x_1, x_2], где x_1 и x_2 имеют размерность head_dim/2.
        let x1 = x.clone().slice([
            0..x.dims()[0], // batch_size
            0..x.dims()[1], // num_heads
            0..x.dims()[2], // seq_len
            0..dim_half,    // первая половина head_dim
        ]);
        let x2 = x.slice([
            0..x.dims()[0],
            0..x.dims()[1],
            0..x.dims()[2],
            dim_half..self.config.dim, // вторая половина head_dim
        ]);

        // Применяем формулу вращения:
        // x_rotated_1 = x_1 * cos - x_2 * sin
        // x_rotated_2 = x_1 * sin + x_2 * cos
        let rotated_x1 = x1.clone().mul(cos_val.clone()).sub(x2.clone().mul(sin_val.clone()));
        let rotated_x2 = x1.mul(sin_val.clone()).add(x2.mul(cos_val.clone()));

        // Конкатенируем преобразованные половины обратно.
        Tensor::cat(vec![rotated_x1, rotated_x2], 3) // Конкатенация по 3-й оси (head_dim).
            .map_err(|e| BurnCoreError::BurnTensor(e)) // Преобразуем ошибку тензора Burn.
    }
}
