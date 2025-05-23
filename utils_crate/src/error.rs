#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

use thiserror::Error;

/// Общий тип ошибки для утилит `utils_crate` и потенциально для всего воркспейса.
///
/// Этот enum агрегирует различные типы ошибок, которые могут возникнуть
/// в утилитарных функциях, предоставляя стандартизированный способ их обработки.
#[derive(Error, Debug)]
pub enum UtilsError {
    /// Ошибка ввода-вывода (I/O).
    ///
    /// Содержит исходную ошибку `std::io::Error` и опционально путь к файлу/директории,
    /// с которым возникла проблема.
    #[error("Ошибка ввода-вывода: {source}")]
    Io {
        /// Исходная ошибка I/O.
        #[from] // Позволяет автоматически конвертировать std::io::Error в UtilsError::Io
        source: std::io::Error,
        /// Опциональный путь, связанный с ошибкой I/O.
        path: Option<String>,
    },

    /// Ошибка сериализации (например, в JSON, TOML).
    ///
    /// Возникает, когда не удается преобразовать структуру данных в строковое представление.
    /// Активируется фичей `serde`.
    #[cfg(feature = "serde")]
    #[error("Ошибка сериализации: {0}")]
    Serialization(String),

    /// Ошибка десериализации (например, из JSON, TOML).
    ///
    /// Возникает, когда не удается преобразовать строковое представление данных
    /// (например, из файла конфигурации) в структуру данных.
    /// Активируется фичей `serde`.
    #[cfg(feature = "serde")]
    #[error("Ошибка десериализации: {0}")]
    Deserialization(String),

    /// Ошибка, связанная с конфигурацией приложения.
    ///
    /// Например, неверный формат файла конфигурации или отсутствующие обязательные поля.
    #[error("Ошибка конфигурации: {0}")]
    Config(String),

    /// Ошибка, указывающая на то, что в утилитарную функцию был передан неверный параметр.
    #[error("Неверный параметр: {0}")]
    InvalidParameter(String),

    /// Ошибка, указывающая, что запрошенная функция или операция не поддерживается.
    #[error("Операция или функция не поддерживается: {0}")]
    NotSupported(String),

    /// Ошибка, указывающая, что внешний ресурс не был найден.
    ///
    /// Например, попытка загрузить модель, которой нет по указанному пути.
    #[error("Ресурс не найден: {0}")]
    ResourceNotFound(String),

    /// Общая ошибка утилиты для случаев, не покрытых другими вариантами.
    ///
    /// Используется, когда ни один из более специфичных вариантов ошибки не подходит.
    #[error("Произошла общая ошибка утилиты: {0}")]
    Generic(String),
}

/// Конвертация из `serde_json::Error` в `UtilsError`.
/// `serde_json::Error` может возникать как при сериализации, так и при десериализации.
/// Эта реализация пытается классифицировать ошибку.
#[cfg(all(feature = "serde", feature = "serde_json"))]
impl From<serde_json::Error> for UtilsError {
    fn from(err: serde_json::Error) -> Self {
        // Проверяем категорию ошибки serde_json для более точного маппинга
        match err.classify() {
            serde_json::error::Category::Io => UtilsError::Io {
                source: err.into(), // serde_json::Error может быть конвертирован в io::Error
                path: None,         // Путь здесь не доступен напрямую из serde_json::Error
            },
            serde_json::error::Category::Syntax | serde_json::error::Category::Data | serde_json::error::Category::Eof => {
                // Эти категории обычно связаны с парсингом (десериализацией)
                UtilsError::Deserialization(format!("Ошибка JSON (десериализация): {}", err))
            }
        }
    }
}

/// Конвертация из `toml::de::Error` (ошибка десериализации TOML) в `UtilsError::Deserialization`.
#[cfg(all(feature = "app_config_serde", feature = "toml"))]
impl From<toml::de::Error> for UtilsError {
    fn from(err: toml::de::Error) -> Self {
        // Оборачиваем ошибку toml в строку и передаем в UtilsError::Config,
        // так как это специфично для загрузки конфигурации.
        // Или можно использовать UtilsError::Deserialization, если считать это общей ошибкой десериализации.
        // Для большей специфичности, если ошибка именно при парсинге AppConfig, лучше UtilsError::Config.
        // Но так как это общая From реализация, UtilsError::Deserialization может быть более универсальным.
        // Выберем Deserialization для консистентности с JSON.
        UtilsError::Deserialization(format!("Ошибка десериализации TOML: {}", err))
    }
}

/// Конвертация из `toml::ser::Error` (ошибка сериализации TOML) в `UtilsError::Serialization`.
#[cfg(all(feature = "app_config_serde", feature = "toml"))]
impl From<toml::ser::Error> for UtilsError {
    fn from(err: toml::ser::Error) -> Self {
        UtilsError::Serialization(format!("Ошибка сериализации TOML: {}", err))
    }
}

impl UtilsError {
    /// Вспомогательный конструктор для создания `UtilsError::Io` с указанием пути.
    ///
    /// # Аргументы
    ///
    /// * `source` - Исходная ошибка `std::io::Error`.
    /// * `path` - Строка или любой тип, который можно преобразовать в `String`, представляющий путь.
    pub fn io_with_path(source: std::io::Error, path: impl Into<String>) -> Self {
        Self::Io {
            source,
            path: Some(path.into()),
        }
    }
}
