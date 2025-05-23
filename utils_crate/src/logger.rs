#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

//! Модуль для инициализации глобального логгера на основе `tracing`.
//!
//! Функциональность этого модуля активируется фичей `logger_utils_feature`.

use tracing::Level;
use tracing_subscriber::{
    fmt,
    EnvFilter,
    layer::SubscriberExt,
    util::SubscriberInitExt,
    Layer,
};

#[cfg(feature = "logger_utils_feature")]
use tracing_appender;

use std::{
    io,
    fs,
    path::Path,
};

use crate::error::UtilsError;

/// Инициализирует глобальный подписчик `tracing`.
///
/// Настраивает вывод в консоль и, опционально, в файл с ротацией.
/// Фильтрует по `RUST_LOG` и явным уровням.
///
/// # Аргументы
/// * `app_name` - Имя приложения (для фильтров и имени файла лога).
/// * `console_level` - Уровень для консоли.
/// * `file_level` - Уровень для файла.
/// * `log_dir` - Опциональная директория для файлов логов.
/// * `_truncate_file_on_start` - **Не используется** с `tracing_appender::rolling::daily`.
///   Оставлен для возможной совместимости или явного указания на неиспользование.
///
/// # Ошибки
/// Возвращает `UtilsError::Generic` при сбое инициализации `tracing`.
/// Проблемы с созданием директории/файла лога логируются как предупреждения,
/// но не приводят к ошибке этой функции, позволяя приложению работать
/// с логированием только в консоль.
///
/// # Паника
/// Может паниковать при неверных директивах `EnvFilter` (например, из-за невалидного
/// `app_name` или уровней логирования после преобразования в строку).
/// Это считается критической ошибкой конфигурации логгера.
#[allow(clippy::module_name_repetitions)] // Имя функции init_tracing_logger в модуле logger - это нормально
#[allow(clippy::unwrap_used, clippy::expect_used)] // Разрешено для инициализации логгера
pub fn init_tracing_logger(
    app_name: &str,
    console_level: Level,
    file_level: Level,
    log_dir: Option<&Path>,
    _truncate_file_on_start: bool,
) -> Result<(), UtilsError> {
    let base_env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Заменяем дефисы на подчеркивания в имени приложения для EnvFilter
    let sanitized_app_name = app_name.replace('-', "_");

    // Фильтр для консоли: базовый env_filter + явный уровень для текущего приложения.
    let console_filter = base_env_filter
        .clone()
        .add_directive(
            format!("{}={}", sanitized_app_name, console_level)
                .parse()
                .expect("Внутренняя ошибка: Неверная директива уровня логирования для консоли"),
        );

    // Слой для вывода в консоль (stderr).
    let console_layer = fmt::layer()
        .with_writer(io::stderr) // Пишем в stderr
        .with_ansi(true) // Включаем ANSI цвета для лучшей читаемости
        .pretty() // Используем "красивый" формат вывода
        .with_filter(console_filter); // Применяем консольный фильтр

    // Вектор для хранения слоев логирования.
    let mut layers: Vec<Box<dyn Layer<_> + Send + Sync + 'static>> = Vec::new();
    layers.push(console_layer.boxed());

    // Настройка логирования в файл, если указана директория и включена фича.
    #[cfg(feature = "logger_utils_feature")]
    if let Some(dir) = log_dir {
        if let Err(e) = fs::create_dir_all(dir) {
            // Если не удалось создать директорию, выводим предупреждение.
            // Используем eprintln!, так как tracing еще не инициализирован.
            eprintln!(
                "[ПРЕДУПРЕЖДЕНИЕ] Не удалось создать директорию логов {:?}: {}. Логирование в файл будет отключено.",
                dir, e
            );
        } else {
            // Настраиваем daily rolling file appender.
            let file_appender =
                tracing_appender::rolling::daily(dir, format!("{}.log", app_name));

            // Фильтр для файла: базовый env_filter + явный уровень для текущего приложения.
            let file_filter = base_env_filter.add_directive( // Используем base_env_filter, а не console_filter
                format!("{}={}", sanitized_app_name, file_level)
                    .parse()
                    .expect("Внутренняя ошибка: Неверная директива уровня логирования для файла"),
            );

            // Слой для вывода в файл.
            let file_layer = fmt::layer()
                .with_writer(file_appender) // Пишем в ротируемый файл
                .with_ansi(false) // Выключаем ANSI цвета для файлов
                .with_filter(file_filter); // Применяем файловый фильтр
            layers.push(file_layer.boxed());
        }
    }

    // Инициализируем глобальный подписчик `tracing` с собранными слоями.
    tracing_subscriber::registry()
        .with(layers)
        .try_init()
        .map_err(|e| UtilsError::Generic(format!("Не удалось инициализировать логгер: {}", e)))?;

    // Информационное сообщение о статусе инициализации логгера
    if log_dir.is_some() && cfg!(feature = "logger_utils_feature") {
        // Проверяем, существует ли директория, чтобы дать более точное сообщение
        if log_dir.unwrap().exists() { // unwrap безопасен, т.к. is_some() проверено
             tracing::info!(
                "Логгер инициализирован. Уровень консоли: {}. Логирование в файл: {:?} (уровень {}).",
                console_level,
                log_dir.unwrap(),
                file_level
            );
        } else {
            // Это сообщение будет выведено, если fs::create_dir_all не удалось,
            // но мы все равно дошли до этой точки (т.е. file_layer не был добавлен).
             tracing::warn!(
                "Логгер инициализирован. Уровень консоли: {}. Попытка логирования в файл в {:?} не удалась (директория не доступна).",
                console_level,
                log_dir.unwrap()
            );
        }
    } else {
         tracing::info!("Логгер инициализирован. Только вывод в консоль (уровень {}).", console_level);
    }
    Ok(())
}
