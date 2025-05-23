use std::error::Error; // Стандартный трейт для ошибок
use std::fmt; // Для форматирования

use thiserror::Error; // Макрос для упрощенного создания типов ошибок

/// Кастомные типы ошибок для крейta `model_loader`.
///
/// Это перечисление инкапсулирует различные ошибки, которые могут возникнуть
/// во время парсинга конфигурации модели, файлового ввода/вывода и загрузки весов.
#[derive(Error, Debug)]
pub enum ModelLoaderError {
    /// Ошибка, указывающая на сбой в операциях файлового ввода/вывода.
    #[error("Ошибка ввода/вывода по пути '{path}': {source}")]
    Io {
        /// Путь к файлу, который вызвал ошибку ввода/вывода.
        path: String,
        /// Исходная ошибка `std::io::Error`.
        #[source]
        source: std::io::Error,
    },

    /// Ошибка, указывающая на сбой во время парсинга файла конфигурации (например, синтаксическая ошибка JSON).
    #[error("Не удалось разобрать файл конфигурации '{path}': {source}")]
    ConfigParsing {
        /// Путь к файлу конфигурации, который не удалось разобрать.
        path: String,
        /// Исходная ошибка парсинга, упакованная в Box для использования как типаж (trait object).
        #[source]
        source: Box<dyn Error + Send + Sync + 'static>,
    },

    /// Ошибка, указывающая на сбой при загрузке записи модели (например, ошибка парсинга safetensors).
    #[error("Не удалось загрузить запись модели из '{path}': {source}")]
    RecordLoading {
        /// Путь к файлу записи модели, который не удалось загрузить.
        path: String,
        /// Исходная ошибка загрузки записи, упакованная в Box.
        #[source]
        source: Box<dyn Error + Send + Sync + 'static>,
    },

    /// Ошибка, указывающая на невалидную или неподдерживаемую конфигурацию модели.
    #[error("Невалидная конфигурация модели: {message}")]
    InvalidConfig {
        /// Описательное сообщение о том, почему конфигурация невалидна.
        message: String,
    },

    /// Общая ошибка для других непредвиденных ситуаций.
    #[error("Произошла непредвиденная ошибка: {message}")]
    Unexpected {
        /// Описательное сообщение для непредвиденной ошибки.
        message: String,
    },
}

// Пример того, как создать ошибку с источником Box<dyn Error>
// Это преобразование позволяет легко конвертировать ошибки serde_json в ModelLoaderError.
impl From<serde_json::Error> for ModelLoaderError {
    fn from(err: serde_json::Error) -> Self {
        ModelLoaderError::ConfigParsing {
            path: "unknown".to_string(), // Заглушка, должна быть заполнена вызывающей стороной с корректным путем
            source: Box::new(err),
        }
    }
}

// Пример того, как создать ошибку из burn::record::Error
// Позволяет конвертировать ошибки загрузки записи (весов) из Burn.
impl From<burn::record::Error> for ModelLoaderError {
    fn from(err: burn::record::Error) -> Self {
        ModelLoaderError::RecordLoading {
            path: "unknown".to_string(), // Заглушка, должна быть заполнена вызывающей стороной
            source: Box::new(err),
        }
    }
}

// Условная компиляция: если включена фича `with_utils_crate_errors`,
// добавляется возможность конвертировать ошибки из `utils_crate`.
#[cfg(feature = "with_utils_crate_errors")]
impl From<utils_crate::errors::ModelHubError> for ModelLoaderError {
    fn from(err: utils_crate::errors::ModelHubError) -> Self {
        ModelLoaderError::Unexpected {
            message: format!("Ошибка из utils_crate: {}", err),
        }
    }
}
