// core_burn/src/error.rs

#![warn(missing_docs, clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code, clippy::unwrap_used, clippy::expect_used)]

// Импортируем стандартные типы ошибок из фреймворка Burn.
use burn::{
    backend::error::Error as BurnBackendError, // Ошибка, специфичная для бэкенда Burn.
    module::Error as BurnModuleError,          // Ошибка, возникающая при работе с модулями Burn.
    record::Error as BurnRecordError,          // Ошибка при сериализации/десериализации (Record).
    tensor::error::Error as BurnTensorError,   // Ошибка при операциях с тензорами Burn.
};

// Импортируем ошибку из нашего модуля KV-кэша.
use crate::kv_cache::KVCacheError;

// Условная компиляция: если активирована фича `with_utils_crate`,
// тогда импортируем и используем ошибку из `utils_crate`.
#[cfg(feature = "with_utils_crate")]
use utils_crate::error::UtilsError;

/// Перечисление всех возможных ошибок, которые могут возникнуть в крейте `core_burn`.
///
/// Этот `enum` агрегирует как специфичные ошибки этого крейта, так и ошибки
/// из зависимостей (например, из Burn или `utils_crate`), предоставляя единый
/// тип ошибки для удобства обработки вызывающим кодом.
#[derive(thiserror::Error, Debug)] // Используем `thiserror` для автоматической генерации трейтов Error и Display.
pub enum BurnCoreError {
    /// Ошибка, связанная с некорректной конфигурацией модели или ее компонентов.
    /// Например, неверные размерности, несовместимые параметры.
    #[error("Некорректная конфигурация: {0}")]
    InvalidConfig(String),

    /// Ошибка, возникшая в модуле Burn (например, при инициализации слоя).
    #[error("Ошибка модуля Burn: {0}")]
    BurnModule(#[from] BurnModuleError), // #[from] позволяет автоматически конвертировать BurnModuleError.

    /// Ошибка, возникшая при работе с Record-структурами Burn (например, при загрузке/сохранении весов).
    #[error("Ошибка записи/чтения Burn (Record): {0}")]
    BurnRecord(#[from] BurnRecordError),

    /// Ошибка, возникшая при операциях с тензорами Burn.
    #[error("Ошибка тензора Burn: {0}")]
    BurnTensor(#[from] BurnTensorError),

    /// Ошибка, специфичная для используемого бэкенда Burn (например, NdArray, WGPU, Tch).
    #[error("Ошибка бэкенда Burn: {0}")]
    BurnBackend(#[from] BurnBackendError),

    /// Ошибка, указывающая на несовместимые размеры тензоров при операциях.
    #[error("Несовместимые размеры или форма тензора: {0}")]
    IncompatibleShape(String),

    /// Ошибка, возникшая при работе с KV-кэшем.
    #[error("Ошибка KV-кэша: {0}")]
    KVCache(#[from] KVCacheError),

    /// Ошибка, возникшая во вспомогательном крейте `utils_crate`.
    /// Этот вариант доступен только если активирована фича `with_utils_crate`.
    #[cfg(feature = "with_utils_crate")]
    #[error("Ошибка из utils_crate: {0}")]
    Utils(#[from] UtilsError),

    /// Общая или неуточненная ошибка в `core_burn`.
    /// Следует использовать с осторожностью, предпочитая более специфичные варианты ошибок.
    #[error("Общая ошибка Core Burn: {0}")]
    Generic(String),
}
