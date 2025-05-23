#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

//! `utils_crate` предоставляет общие структуры данных, обработку ошибок и общие утилиты
//! для проекта AI Code Assistant.

// --- Модуль для общих ошибок ---
pub mod error;
pub use error::UtilsError;

// --- Модуль для параметров сэмплинга ---
pub mod sampling_params;
pub use sampling_params::{GenerationConstraint, SamplingParams, StopTokens};

// --- Модуль для представления задач инференса и их результатов ---
pub mod inference_task;
pub use inference_task::{
    ChatMessage, InferenceResponse, InferenceTask, MessageContent, TaskInput, UsageMetrics,
};

// --- Ваши утилитарные модули (управляются фичами) ---

#[cfg(feature = "path_utils_feature")]
pub mod path; // Имя модуля соответствует имени файла src/path.rs
#[cfg(feature = "path_utils_feature")]
pub use path::{ensure_dir_exists, find_file_in_dir, sanitize_path_component};

#[cfg(feature = "logger_utils_feature")]
pub mod logger;
#[cfg(feature = "logger_utils_feature")]
pub use logger::init_tracing_logger;

#[cfg(feature = "app_config_serde")] // Эта фича включает и toml и serde для AppConfig
pub mod config;
#[cfg(feature = "app_config_serde")]
pub use config::AppConfig;
