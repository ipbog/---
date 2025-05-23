#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

use tracing::Level;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

#[cfg(feature = "logger_utils_feature")]
use tracing_appender;

use std::{
    fs, io,
    path::Path,
};

use crate::error::UtilsError;

/// Инициализирует глобальный `tracing` логгер.
///
/// Эта функция настраивает подписчика `tracing`, который:
/// - Фильтрует логи на основе переменной окружения `RUST_LOG`.
/// - Выводит логи в человекочитаемом формате в `stderr` (консоль) с указанным уровнем.
/// - Опционально выводит логи в файл с указанным уровнем, используя ежедневную ротацию.
///
/// # Arguments
/// * `app_name` - Имя приложения, используется для именования файлов логов.
/// * `console_level` - Максимальный уровень логов для вывода в консоль.
/// * `file_level` - Максимальный уровень логов для вывода в файл.
/// * `log_dir` - Опциональный путь к директории, куда будут записываться файлы логов.
/// * `_truncate_file_on_start` - Этот параметр не используется с ротируемыми файловыми аппендерами,
/// но сохранен для совместимости API, если потребуется для других типов аппендеров в будущем.
///
/// # Errors
/// Возвращает `UtilsError::Io`, если есть проблема с созданием директории или файла логов.
/// Возвращает `UtilsError::Generic`, если инициализация логгера не удалась.
#[allow(clippy::module_name_repetitions)] // Имя функции init_tracing_logger достаточно ясно.
#[allow(clippy::unwrap_used, clippy::expect_used)] // Разрешено для инициализации логгера; паника при настройке логгера приемлема.
pub fn init_tracing_logger(
    app_name: &str,
    console_level: Level,
    file_level: Level,
    log_dir: Option<&Path>,
    _truncate_file_on_start: bool, // Параметр сейчас не используется с rolling appender
) -> Result<(), UtilsError> {
    let base_env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Фильтр для консоли: общий env_filter + уровень для текущего приложения
    let console_filter = base_env_filter
        .clone()
        .add_directive(
            format!("{}={}", app_name.replace('-', "_"), console_level)
                .parse()
                .expect("Invalid console log level directive"),
        );

    let console_layer = fmt::layer()
        .with_writer(io::stderr)
        .with_ansi(true)
        .pretty()
        .with_filter(console_filter);

    let mut layers = Vec::new();
    let console_boxed_layer = console_layer.boxed(); // Box layer to store in Vec
    layers.push(console_boxed_layer);

    #[cfg(feature = "logger_utils_feature")]
    if let Some(dir) = log_dir {
        if let Err(e) = fs::create_dir_all(dir) {
            // Логируем предупреждение через стандартный `eprintln`,
            // так как глобальный логгер еще не инициализирован.
            eprintln!(
                "[WARN] Failed to create log directory {:?}: {}. File logging disabled.",
                dir, e
            );
        } else {
            let file_appender = tracing_appender::rolling::daily(dir, format!("{}.log", app_name));

            // Фильтр для файла: общий env_filter + уровень для текущего приложения
            let file_filter = base_env_filter.add_directive(
                format!("{}={}", app_name.replace('-', "_"), file_level)
                    .parse()
                    .expect("Invalid file log level directive"),
            );

            let file_layer = fmt::layer()
                .with_writer(file_appender)
                .with_ansi(false) // Обычно ANSI не нужен для файлов
                .with_filter(file_filter);
            layers.push(file_layer.boxed()); // Box layer
        }
    }

    tracing_subscriber::registry()
        .with(layers) // Используем собранные слои
        .try_init()
        .map_err(|e| UtilsError::Generic(format!("Failed to initialize logger: {}", e)))?;

    // Теперь можно использовать tracing::info
    if log_dir.is_some() && cfg!(feature = "logger_utils_feature") {
        tracing::info!(
            "Logger initialized with console and file output to directory: {:?}",
            log_dir.unwrap() // Безопасно, так как log_dir.is_some()
        );
    } else {
        tracing::info!("Logger initialized with console output only.");
    }
    Ok(())
}
