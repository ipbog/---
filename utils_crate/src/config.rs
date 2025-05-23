#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

//! Модуль для загрузки и управления конфигурацией приложения.
//!
//! Этот модуль определяет структуру `AppConfig` и связанные с ней под-конфигурации,
//! которые могут быть десериализованы из TOML-файла.
//! Функциональность этого модуля активируется фичей `app_config_serde`.

#[cfg(feature = "app_config_serde")]
use serde::Deserialize;
#[cfg(feature = "app_config_serde")]
use std::path::Path;
#[cfg(feature = "app_config_serde")]
use crate::error::UtilsError;
#[cfg(feature = "app_config_serde")]
use tracing::warn;

/// Глобальная конфигурация приложения.
///
/// Эта структура агрегирует все специфичные конфигурационные секции.
/// При десериализации из TOML, если какая-либо секция отсутствует,
/// для нее будут использованы значения по умолчанию.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "app_config_serde", derive(Deserialize))]
pub struct AppConfig {
    /// Конфигурация, связанная с моделями.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_model_config"))]
    pub model_config: ModelConfigSub,

    /// Конфигурация API сервера.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_api_config"))]
    pub api_config: ApiConfigSub,

    /// Конфигурация логирования.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_logging_config"))]
    pub logging_config: LoggingConfigSub,

    /// Общие настройки приложения.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_general_settings"))]
    pub general_settings: GeneralSettingsSub,
}

// Функции для предоставления значений по умолчанию для полей AppConfig при десериализации.
#[cfg(feature = "app_config_serde")]
fn default_model_config() -> ModelConfigSub {
    ModelConfigSub::default()
}
#[cfg(feature = "app_config_serde")]
fn default_api_config() -> ApiConfigSub {
    ApiConfigSub::default()
}
#[cfg(feature = "app_config_serde")]
fn default_logging_config() -> LoggingConfigSub {
    LoggingConfigSub::default()
}
#[cfg(feature = "app_config_serde")]
fn default_general_settings() -> GeneralSettingsSub {
    GeneralSettingsSub::default()
}

impl Default for AppConfig {
    /// Возвращает конфигурацию `AppConfig` со значениями по умолчанию для всех полей.
    fn default() -> Self {
        Self {
            model_config: ModelConfigSub::default(),
            api_config: ApiConfigSub::default(),
            logging_config: LoggingConfigSub::default(),
            general_settings: GeneralSettingsSub::default(),
        }
    }
}

/// Конфигурация, связанная с моделями.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "app_config_serde", derive(Deserialize))]
pub struct ModelConfigSub {
    /// Путь к директории, где хранятся или кэшируются модели.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_models_dir_path_app"))]
    pub model_dir: String,
    /// Список моделей для предзагрузки при старте.
    #[cfg_attr(feature = "app_config_serde", serde(default))]
    pub preload_models: Vec<String>,
    /// Разрешить динамическую загрузку моделей.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_allow_dynamic_loading_app"))]
    pub allow_dynamic_loading: bool,
}

#[cfg(feature = "app_config_serde")]
fn default_models_dir_path_app() -> String {
    "./.model_cache".to_string()
}
#[cfg(feature = "app_config_serde")]
fn default_allow_dynamic_loading_app() -> bool {
    true
}

impl Default for ModelConfigSub {
    fn default() -> Self {
        Self {
            model_dir: default_models_dir_path_app(),
            preload_models: Vec::new(),
            allow_dynamic_loading: default_allow_dynamic_loading_app(),
        }
    }
}

/// Конфигурация API сервера.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "app_config_serde", derive(Deserialize))]
pub struct ApiConfigSub {
    /// Хост API сервера.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_api_host_app"))]
    pub host: String,
    /// Порт API сервера.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_api_port_app"))]
    pub port: u16,
    /// Включена ли поддержка CORS.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_api_cors_enabled_app"))]
    pub cors_enabled: bool,
}

#[cfg(feature = "app_config_serde")]
fn default_api_host_app() -> String {
    "127.0.0.1".to_string()
}
#[cfg(feature = "app_config_serde")]
fn default_api_port_app() -> u16 {
    8080
}
#[cfg(feature = "app_config_serde")]
fn default_api_cors_enabled_app() -> bool {
    false
}

impl Default for ApiConfigSub {
    fn default() -> Self {
        Self {
            host: default_api_host_app(),
            port: default_api_port_app(),
            cors_enabled: default_api_cors_enabled_app(),
        }
    }
}

/// Конфигурация логирования.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "app_config_serde", derive(Deserialize))]
pub struct LoggingConfigSub {
    /// Уровень логирования (например, "info", "debug").
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_log_level_app"))]
    pub level: String,
    /// Опциональный путь к файлу логов.
    #[cfg_attr(feature = "app_config_serde", serde(default))]
    pub log_file: Option<String>,
    /// Использовать ли JSON формат для логов.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_log_json_app"))]
    pub json_format: bool,
}

#[cfg(feature = "app_config_serde")]
fn default_log_level_app() -> String {
    "info".to_string()
}
#[cfg(feature = "app_config_serde")]
fn default_log_json_app() -> bool {
    false
}

impl Default for LoggingConfigSub {
    fn default() -> Self {
        Self {
            level: default_log_level_app(),
            log_file: None,
            json_format: default_log_json_app(),
        }
    }
}

/// Общие настройки приложения.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "app_config_serde", derive(Deserialize))]
pub struct GeneralSettingsSub {
    /// Тема интерфейса.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_theme_app"))]
    pub theme: String,
    /// Язык интерфейса.
    #[cfg_attr(feature = "app_config_serde", serde(default = "default_language_app"))]
    pub language: String,
}

#[cfg(feature = "app_config_serde")]
fn default_theme_app() -> String {
    "dark".to_string()
}
#[cfg(feature = "app_config_serde")]
fn default_language_app() -> String {
    "en_US".to_string()
}

impl Default for GeneralSettingsSub {
    fn default() -> Self {
        Self {
            theme: default_theme_app(),
            language: default_language_app(),
        }
    }
}

#[cfg(feature = "app_config_serde")]
impl AppConfig {
    /// Загружает конфигурацию приложения из TOML-файла.
    ///
    /// Если файл по указанному пути не найден, будет возвращена конфигурация
    /// со значениями по умолчанию, и в лог будет выведено предупреждение.
    ///
    /// # Аргументы
    ///
    /// * `file_path` - Путь к TOML-файлу конфигурации.
    ///
    /// # Ошибки
    ///
    /// Возвращает `UtilsError::Io` при ошибках чтения файла.
    /// Возвращает `UtilsError::Deserialization` при ошибках парсинга TOML
    /// (через `From<toml::de::Error> for UtilsError`).
    /// Ошибку `UtilsError::Config` можно использовать, если нужна более специфичная
    /// ошибка именно конфигурации, но `From` реализация для `toml::de::Error`
    /// уже мапит на `UtilsError::Deserialization`.
    pub fn load_from_toml(file_path: &Path) -> Result<Self, UtilsError> {
        if !file_path.exists() {
            warn!(
                "Файл AppConfig не найден по пути {:?}, используется конфигурация по умолчанию.",
                file_path
            );
            return Ok(Self::default());
        }

        let config_str = std::fs::read_to_string(file_path)?; // Автоматически конвертируется в UtilsError::Io

        // toml::from_str вернет toml::de::Error, который конвертируется в UtilsError::Deserialization
        // через реализацию From<toml::de::Error> for UtilsError.
        // Если нужно явно указать, что это ошибка конфигурации, можно сделать так:
        // toml::from_str(&config_str).map_err(|e| UtilsError::Config(format!("Ошибка парсинга TOML: {}", e)))
        // Но использование From более идиоматично, если оно уже есть.
        toml::from_str(&config_str).map_err(UtilsError::from)
    }
}
