// core_burn/src/lib.rs

// Включаем строгие правила линтинга для всего крейта.
#![warn(
    missing_docs, // Предупреждать об отсутствующей документации для публичных элементов.
    clippy::all, // Все стандартные проверки Clippy.
    clippy::pedantic, // Более строгие ("педантичные") проверки Clippy.
    clippy::nursery // Экспериментальные проверки Clippy (могут быть нестабильны).
)]
// Запрещаем использование небезопасных конструкций и потенциально проблемных методов.
#![deny(
    unsafe_code, // Запрет `unsafe` блоков без явного `allow`.
    clippy::unwrap_used, // Запрет использования `.unwrap()`.
    clippy::expect_used // Запрет использования `.expect()`.
)]

//! # `core_burn`
//!
//! Этот крейт (`core_burn`) является ядром для реализации моделей машинного обучения
//! с использованием фреймворка [Burn](https://burn.dev/). Он предоставляет определения
//! архитектур нейронных сетей, таких как Gemma, а также вспомогательные компоненты,
//! необходимые для их работы: Rotary Positional Embeddings (RoPE) и Key-Value кэш (KV-кэш).
//!
//! ## Назначение
//!
//! Основная цель крейта — служить фундаментом для более высокоуровневых систем,
//! таких как движок инференса (`inference_engine`) или движок обучения (`training_engine`),
//! предоставляя им безопасные, производительные и гибкие реализации моделей.
//!
//! ## Структура
//!
//! Крейт организован следующим образом:
//! - `architectures`: Содержит определения различных архитектур моделей (например, `gemma`).
//! - `error`: Определяет кастомные типы ошибок для этого крейта.
//! - `rope`: Реализация Rotary Positional Embeddings.
//! - `kv_cache`: Реализация механизма KV-кэширования для моделей-трансформеров.

// Объявляем публичные модули, входящие в состав крейта.
pub mod architectures;
pub mod error;
pub mod kv_cache;
pub mod rope;

// Реэкспортируем наиболее важные и часто используемые элементы из модулей
// для удобства их использования потребителями этого крейта.

// Ошибки
pub use error::BurnCoreError;

// Компоненты моделей
pub use kv_cache::{KVCache, KVCacheError, MhaCache}; // MhaCache из burn реэкспортируется через наш kv_cache
pub use rope::{RotaryPositionalEmbedding, RotaryPositionalEmbeddingConfig};

// Архитектуры и их конфигурации
pub use architectures::gemma::{
    // Основная модель Gemma и ее компоненты
    GemmaModel,
    GemmaDecoderBlock,
    GemmaAttention,
    GemmaFeedForward,
    // Конфигурации для модели Gemma и ее компонентов
    GemmaModelConfig,
    GemmaDecoderBlockConfig,
    GemmaAttentionConfig,
    GemmaFeedForwardConfig,
    // Record-структуры для сериализации/десериализации весов
    GemmaModelRecord,
    GemmaDecoderBlockRecord,
    GemmaAttentionRecord, // Реэкспортируем все рекорды для полноты API
    GemmaFeedForwardRecord,
};
pub use architectures::ModelInfo; // Общая информация о модели (тип, размеры и т.д.)
