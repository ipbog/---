#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

//! Модуль определяет структуры для управления параметрами семплирования
//! при генерации текста языковыми моделями.

#[cfg(feature = "inference_types_serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(all(feature = "inference_types_serde", feature = "serde_json"))]
use serde_json::Value as JsonValue;

/// Ограничение для управления процессом генерации токенов.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub enum GenerationConstraint {
    /// Отсутствие ограничений.
    #[default]
    None,
    /// Ограничить генерацию регулярным выражением.
    Regex(String),
    /// Ограничить генерацию грамматикой Lark.
    Lark(String),
    /// Ограничить генерацию JSON-схемой.
    /// Требует фичи `inference_types_serde` и `serde_json`.
    #[cfg(all(feature = "inference_types_serde", feature = "serde_json"))]
    JsonSchema(JsonValue),
}

/// Последовательности для остановки генерации.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub enum StopTokens {
    /// Строковые последовательности.
    Seqs(Vec<String>),
    /// Идентификаторы токенов.
    Ids(Vec<u32>),
    /// Не определены.
    None,
}

impl Default for StopTokens {
    fn default() -> Self {
        StopTokens::None
    }
}

/// Параметры для управления процессом семплирования.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub struct SamplingParams {
    /// Температура семплирования.
    pub temperature: Option<f64>,
    /// Вероятность для nucleus sampling (top-P).
    pub top_p: Option<f64>,
    /// Top-K семплирование.
    pub top_k: Option<usize>,
    /// Минимальная вероятность токена (min-P).
    pub min_p: Option<f64>,
    /// Штраф за повторение токенов.
    pub repeat_penalty: Option<f32>,
    /// Штраф за присутствие токенов.
    pub presence_penalty: Option<f32>,
    /// Смещение для логитов токенов.
    pub logits_bias: Option<HashMap<u32, f32>>,
    /// Количество возвращаемых top_n логарифмов вероятностей.
    pub top_n_logprobs: usize,
    /// Начальное значение (seed) для генератора случайных чисел.
    pub seed: Option<u64>,
    /// Ограничение генерации.
    #[cfg_attr(feature = "inference_types_serde", serde(default))]
    pub constraint: GenerationConstraint,
}

impl Default for SamplingParams {
    /// Значения по умолчанию для `SamplingParams`.
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
    /// Параметры для детерминированной (жадной) генерации.
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
