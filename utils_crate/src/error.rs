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

/// Общий тип ошибки для утилит и потенциально для всего рабочего пространства.
#[derive(Error, Debug)]
pub enum UtilsError {
    /// Произошла ошибка ввода-вывода.
    #[error("I/O error: {source}")]
    Io {
        #[from] // Позволяет автоматически конвертировать std::io::Error
        source: std::io::Error,
        /// Опциональный контекст пути для ошибки ввода-вывода.
        path: Option<String>,
    },

    /// Ошибка сериализации (например, в JSON, TOML).
    #[cfg(feature = "serde")]
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Ошибка десериализации (например, из JSON, TOML).
    #[cfg(feature = "serde")]
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Ошибка, связанная с конфигурацией приложения.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Ошибка, указывающая, что в утилитарную функцию был передан неверный параметр.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Ошибка, указывающая, что запрошенная функция или операция не поддерживается.
    #[error("Operation or feature not supported: {0}")]
    NotSupported(String),

    /// Ошибка, указывающая, что внешний ресурс не найден.
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// Общая ошибка утилиты для случаев, не покрытых другими вариантами.
    #[error("A generic utility error occurred: {0}")]
    Generic(String),
}

// Реализации From для удобства, если фичи включены

#[cfg(all(feature = "serde", feature = "serde_json"))]
impl From<serde_json::Error> for UtilsError {
    fn from(err: serde_json::Error) -> Self {
        // serde_json::Error может возникать как при сериализации, так и при десериализации.
        // Мы не можем точно знать контекст здесь, но Deserialization более частый случай для from_str/from_reader.
        // Если ошибка возникла при сериализации, лучше использовать .map_err(|e| UtilsError::Serialization(...))
        UtilsError::Deserialization(format!("JSON error: {}", err))
    }
}

#[cfg(all(feature = "app_config_serde", feature = "toml"))]
impl From<toml::de::Error> for UtilsError {
    fn from(err: toml::de::Error) -> Self {
        UtilsError::Deserialization(format!("TOML deserialization error: {}", err))
    }
}

#[cfg(all(feature = "app_config_serde", feature = "toml"))]
impl From<toml::ser::Error> for UtilsError {
    fn from(err: toml::ser::Error) -> Self {
        UtilsError::Serialization(format!("TOML serialization error: {}", err))
    }
}

// Вспомогательный конструктор для I/O ошибок с путем
impl UtilsError {
    /// Создает ошибку `Io` с указанием пути.
    pub fn io_with_path(source: std::io::Error, path: impl Into<String>) -> Self {
        Self::Io {
            source,
            path: Some(path.into()),
        }
    }
}
