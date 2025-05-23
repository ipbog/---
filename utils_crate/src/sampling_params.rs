#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

#[cfg(feature = "inference_types_serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "serde_json")]
use serde_json::Value as JsonValue;

/// Ограничение для управления генерацией токенов.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub enum GenerationConstraint {
    /// Нет ограничений.
    #[default]
    None,
    /// Ограничить генерацию на соответствие регулярному выражению.
    Regex(String),
    /// Ограничить генерацию на следование грамматике Lark.
    Lark(String),
    /// Ограничить генерацию на создание валидного JSON согласно схеме.
    #[cfg(feature = "serde_json")]
    JsonSchema(JsonValue),
}

/// Последовательности или ID токенов для остановки генерации.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub enum StopTokens {
    /// Остановить генерацию при встрече любой из этих строковых последовательностей.
    Seqs(Vec<String>),
    /// Остановить генерацию при встрече любого из этих ID токенов.
    Ids(Vec<u32>),
    /// Стоп-токены не определены.
    None,
}

impl Default for StopTokens {
    fn default() -> Self {
        StopTokens::None
    }
}

/// Параметры для управления процессом сэмплирования во время генерации токенов.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub struct SamplingParams {
    /// Температура для сэмплирования, контролирует случайность. Более высокие значения означают большую случайность.
    /// Диапазон: (0.0, бесконечность). По умолчанию: 0.7
    pub temperature: Option<f64>,
    /// Вероятность для nucleus sampling. Рассматриваются только токены с кумулятивной вероятностью выше этого значения.
    /// Диапазон: (0.0, 1.0]. По умолчанию: 0.9
    pub top_p: Option<f64>,
    /// Top-K sampling: рассматриваются только K наиболее вероятных следующих токенов.
    /// Диапазон: [0, бесконечность). По умолчанию: 50 (отключено, если 0).
    pub top_k: Option<usize>,
    /// Минимальная вероятность для рассмотрения токена.
    /// Диапазон: [0.0, 1.0]. По умолчанию: None.
    pub min_p: Option<f64>,
    /// Штраф за повторение токенов. Более высокие значения препятствуют повторению.
    /// По умолчанию: 1.1 (небольшой штраф).
    pub repeat_penalty: Option<f32>,
    /// Штраф за токены, которые уже появлялись.
    /// По умолчанию: None (нет штрафа за присутствие).
    pub presence_penalty: Option<f32>,
    /// Явное смещение для логитов определенных токенов. Отображает ID токена на значение смещения.
    /// По умолчанию: None.
    pub logits_bias: Option<HashMap<u32, f32>>,
    /// Количество возвращаемых верхних лог-вероятностей.
    /// По умолчанию: 0 (лог-вероятности не возвращаются).
    pub top_n_logprobs: usize,
    /// Сид для генератора случайных чисел для обеспечения воспроизводимости.
    /// По умолчанию: 42.
    pub seed: Option<u64>,
    /// Ограничение, применяемое во время генерации.
    /// По умолчанию: `GenerationConstraint::None`.
    #[cfg_attr(feature = "inference_types_serde", serde(default))]
    pub constraint: GenerationConstraint,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(50),
            min_p: None,
            repeat_penalty: Some(1.1),
            presence_penalty: None,
            logits_bias: None,
            top_n_logprobs: 0,
            seed: Some(42),
            constraint: GenerationConstraint::default(),
        }
    }
}

impl SamplingParams {
    /// Возвращает параметры сэмплирования, настроенные для детерминированной (жадной) генерации.
    pub fn deterministic() -> Self {
        Self {
            temperature: Some(0.0),
            top_k: Some(1),
            top_p: None,
            min_p: None,
            ..Default::default()
        }
    }
}
