#![warn(
    missing_docs, // Предупреждать, если публичные элементы не документированы.
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used, // Предупреждать об использовании .unwrap()
    clippy::expect_used  // Предупреждать об использовании .expect()
)]
#![deny(
    unsafe_code,        // Запретить использование unsafe блоков.
    unused_mut,         // Запретить неиспользуемые изменяемые переменные.
    unused_imports,     // Запретить неиспользуемые импорты.
    unused_attributes   // Запретить неиспользуемые атрибуты.
)]

//! `utils_crate` предоставляет общие структуры данных, обработку ошибок
//! и распространенные утилиты для проекта AI Code Assistant.
//!
//! Этот крейт спроектирован так, чтобы быть модульным, позволяя другим частям
//! проекта выборочно включать необходимую функциональность через систему фич (features).
//!
//! # Основные модули:
//!
//! - [`error`]: Определяет общий тип ошибки `UtilsError` для всего крейта.
//! - [`sampling_params`]: Содержит структуры для настройки параметров семплирования
//!   при генерации текста моделями (`SamplingParams`, `GenerationConstraint`).
//! - [`inference_task`]: Определяет структуры для представления задач инференса
//!   (`InferenceTask`, `TaskInput`, `ChatMessage`) и их результатов (`InferenceResponse`).
//! - [`config`]: (активируется фичей `app_config_serde`) Предоставляет `AppConfig` для
//!   загрузки и управления конфигурацией приложения из TOML-файлов.
//! - [`path`]: (активируется фичей `path_utils_feature`) Утилиты для работы
//!   с путями файловой системы.
//! - [`logger`]: (активируется фичей `logger_utils_feature`) Утилиты для
//!   инициализации системы логирования на базе `tracing`.
//!
//! # Использование фич (Features)
//!
//! Крейт использует фичи для управления зависимостями и включаемой функциональностью.
//! Например, чтобы использовать утилиты для логирования, необходимо включить фичу `logger_utils_feature`
//! в `Cargo.toml` вашего проекта:
//!
//! ```toml
//! # В Cargo.toml вашего проекта
//! # utils_crate = { path = "path/to/utils_crate", features = ["logger_utils_feature"] }
//! ```
//!
//! Фича `default` включает наиболее часто используемый набор утилит.

// --- Модуль для общих ошибок ---
pub mod error;
pub use error::UtilsError; // Реэкспорт для удобства использования. Тип ошибки UtilsError.

// --- Модуль для параметров сэмплинга ---
pub mod sampling_params;
pub use sampling_params::{GenerationConstraint, SamplingParams, StopTokens}; // Реэкспорт.

// --- Модуль для представления задач инференса и их результатов ---
pub mod inference_task;
pub use inference_task::{
    ChatMessage, InferenceResponse, InferenceTask, MessageContent, TaskInput, UsageMetrics,
}; // Реэкспорт.

// --- Утилитарные модули (управляются фичами) ---

/// Модуль с утилитами для работы с путями файловой системы.
///
/// Активируется фичей `path_utils_feature`.
#[cfg(feature = "path_utils_feature")]
pub mod path; // Имя модуля path, файл src/path.rs
#[cfg(feature = "path_utils_feature")]
pub use path::{ensure_dir_exists, find_file_in_dir, sanitize_path_component}; // Реэкспорт.

/// Модуль с утилитами для инициализации логирования.
///
/// Активируется фичей `logger_utils_feature`.
#[cfg(feature = "logger_utils_feature")]
pub mod logger;
#[cfg(feature = "logger_utils_feature")]
pub use logger::init_tracing_logger; // Реэкспорт.

/// Модуль для загрузки и управления конфигурацией приложения.
///
/// Активируется фичей `app_config_serde`.
#[cfg(feature = "app_config_serde")]
pub mod config;
#[cfg(feature = "app_config_serde")]
pub use config::AppConfig; // Реэкспорт.
