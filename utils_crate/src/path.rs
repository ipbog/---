#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

//! Модуль с утилитами для работы с путями файловой системы.
//!
//! Функциональность этого модуля активируется фичей `path_utils_feature`.

use std::fs;
use std::path::{Path, PathBuf};
use crate::error::UtilsError;
use tracing::{debug, error, info};

/// Гарантирует, что директория по указанному пути существует.
/// Если не существует, создает ее (включая родительские).
///
/// # Аргументы
/// * `dir_path` - Путь к директории.
///
/// # Ошибки
/// Возвращает `UtilsError::Io` или `UtilsError::InvalidParameter`.
pub fn ensure_dir_exists(dir_path: &Path) -> Result<(), UtilsError> {
    if !dir_path.exists() {
        info!("Создание директории: {:?}", dir_path);
        fs::create_dir_all(dir_path)
            .map_err(|e| UtilsError::io_with_path(e, dir_path.to_string_lossy().into_owned()))?;
        debug!("Директория успешно создана: {:?}", dir_path);
    } else if !dir_path.is_dir() {
        let err_msg = format!(
            "Путь {:?} существует, но не является директорией.",
            dir_path
        );
        error!("{}", err_msg);
        return Err(UtilsError::InvalidParameter(err_msg));
    } else {
        debug!("Директория уже существует: {:?}", dir_path);
    }
    Ok(())
}

/// Ищет файл с указанным именем в директории (нерекурсивно).
///
/// # Аргументы
/// * `dir` - Директория для поиска.
/// * `file_name` - Имя файла.
///
/// # Возвращает
/// `Ok(Some(PathBuf))` если найден, `Ok(None)` если нет.
///
/// # Ошибки
/// Возвращает `UtilsError::Io` или `UtilsError::InvalidParameter`.
pub fn find_file_in_dir(dir: &Path, file_name: &str) -> Result<Option<PathBuf>, UtilsError> {
    if !dir.is_dir() {
        let err_msg = format!(
            "Указанный путь не является директорией: {:?}",
            dir
        );
        error!("{}", err_msg);
        return Err(UtilsError::InvalidParameter(err_msg));
    }

    debug!("Поиск файла '{}' в директории: {:?}", file_name, dir);
    for entry_result in fs::read_dir(dir).map_err(|e| UtilsError::io_with_path(e, dir.to_string_lossy().into_owned()))? {
        let entry = entry_result.map_err(|e| {
            UtilsError::io_with_path(e, format!("запись в {:?}", dir))
        })?;
        let path = entry.path();
        if path.is_file() && path.file_name().map_or(false, |name| name == file_name) {
            info!("Файл найден: {:?}", path);
            return Ok(Some(path));
        }
    }
    info!("Файл '{}' не найден в директории: {:?}", file_name, dir);
    Ok(None)
}

/// "Очищает" (санитизирует) строку для безопасного использования как компонент пути.
///
/// Заменяет проблемные символы на `_`.
/// Сохраняет ASCII буквы, цифры, `-`, `_`, `.`.
///
/// # Аргументы
/// * `component` - Строка для санитизации.
///
/// # Возвращает
/// Новая `String` с замененными символами.
pub fn sanitize_path_component(component: &str) -> String {
    component
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect()
}
