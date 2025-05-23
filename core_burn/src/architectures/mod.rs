// core_burn/src/architectures/mod.rs

#![warn(missing_docs, clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code, clippy::unwrap_used, clippy::expect_used)]

//! Корневой модуль для определения различных архитектур моделей машинного обучения.
//!
//! Каждая поддерживаемая архитектура (например, Gemma, LLaMA) должна быть представлена
//! в своем подмодуле внутри этого модуля. Также здесь определяются общие типы,
//! используемые для идентификации и конфигурации моделей.

// Подключаем подмодуль для архитектуры Gemma.
pub mod gemma;

// В будущем здесь могут быть добавлены другие архитектуры:
// pub mod llama;
// pub mod phi;

use serde::{Deserialize, Serialize};

/// Перечисление, представляющее тип архитектуры модели.
///
/// Используется для идентификации модели при загрузке конфигурации (например, из `config.json`).
/// Атрибут `#[serde(rename_all = "kebab-case")]` позволяет корректно десериализовать
/// значения типа "gemma", "llama-2" и т.д.
/// Атрибут `#[serde(other)]` обеспечивает устойчивость к неизвестным типам моделей.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ModelType {
    /// Модель семейства Gemma.
    Gemma,
    /// Модель семейства LLaMA.
    Llama,
    /// Модель семейства Phi.
    Phi,
    // Другие известные типы моделей могут быть добавлены сюда.
    /// Представляет любой другой или неизвестный тип модели.
    /// Значение будет захвачено как `String`.
    #[serde(other)]
    Other(String),
}

/// Структура, содержащая общую мета-информацию о модели.
///
/// Эта информация обычно извлекается из файла конфигурации модели (например, `config.json`)
/// и может быть использована для высокоуровневой логики, такой как выбор
/// соответствующего токенизатора или настройка параметров инференса.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// Тип архитектуры модели (например, Gemma, Llama).
    #[serde(rename = "model_type")] // Указывает, что в JSON это поле называется "model_type".
    pub model_type: ModelType,

    /// Максимальная длина последовательности, которую может обработать модель.
    /// Часто совпадает с `max_position_embeddings` из конфигурации модели.
    pub max_sequence_length: usize, // Переименовано для ясности (было max_position_embeddings)

    /// Размер словаря токенизатора.
    pub vocab_size: usize,

    /// Размерность скрытого слоя модели (embedding dimension).
    pub hidden_size: usize,

    /// Общее количество скрытых слоев (декодерных блоков) в модели.
    pub num_hidden_layers: usize,

    /// Количество голов внимания в механизме multi-head attention.
    pub num_attention_heads: usize,

    /// Количество голов для Key и Value в механизме Grouped Query Attention (GQA).
    /// Если GQA не используется, это значение обычно равно `num_attention_heads`.
    pub num_key_value_heads: usize,
    // Другие общие параметры, такие как `rms_norm_eps`, `intermediate_size` и т.д.,
    // могут быть здесь или в более специфичных конфигурациях модели.
}
