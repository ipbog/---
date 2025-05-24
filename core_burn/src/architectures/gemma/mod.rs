// core_burn/src/architectures/gemma/mod.rs

#![warn(missing_docs, clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code, clippy::unwrap_used, clippy::expect_used)]

//! Модуль, инкапсулирующий полную реализацию архитектуры модели Gemma.
//!
//! Он объединяет все компоненты Gemma, такие как механизм внимания (`attention`),
//! полносвязные сети (`ffn`), декодерные блоки и основную структуру модели (`model`).
//! Этот модуль также реэкспортирует ключевые типы для удобства использования.

// Подключаем подмодули, содержащие реализацию отдельных компонентов Gemma.
pub mod attention; // Механизм внимания (Self-Attention)
pub mod ffn;       // Полносвязная сеть (Feed-Forward Network)
pub mod model;     // Основная модель, декодерные блоки, конфигурации

// Реэкспортируем публичные структуры и конфигурации из подмодулей,
// чтобы они были легко доступны через `core_burn::architectures::gemma::*`.

// Компоненты внимания
pub use attention::{GemmaAttention, GemmaAttentionConfig, GemmaAttentionRecord};

// Компоненты полносвязной сети
pub use ffn::{GemmaFeedForward, GemmaFeedForwardConfig, GemmaFeedForwardRecord};

// Основная модель, декодерные блоки и их конфигурации/рекорды
pub use model::{
    GemmaModel,
    GemmaModelConfig,
    GemmaModelRecord,
    GemmaDecoderBlock,
    GemmaDecoderBlockConfig,
    GemmaDecoderBlockRecord,
};
