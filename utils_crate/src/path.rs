#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

use std::fs;
use std::path::{Path, PathBuf};
use crate::error::UtilsError;
use tracing::{debug, error, info};

/// Гарантирует, что директория существует, создавая ее при необходимости.
///
/// # Arguments
/// * `dir_path` - Путь к директории, существование которой нужно обеспечить.
///
/// # Errors
/// Возвращает `UtilsError::Io`, если директория не может быть создана или если по указанному пути существует файл.
/// Возвращает `UtilsError::InvalidParameter`, если предоставленный путь не является директорией.
pub fn ensure_dir_exists(dir_path: &Path) -> Result<(), UtilsError> {
    if !dir_path.exists() {
        info!("Creating directory: {:?}", dir_path);
        fs::create_dir_all(dir_path).map_err(|e| {
            UtilsError::io_with_path(e, dir_path.to_string_lossy().into_owned())
        })?;
        debug!("Directory created: {:?}", dir_path);
    } else if !dir_path.is_dir() {
        let err_msg = format!("Path {:?} exists but is not a directory.", dir_path);
        error!("{}", err_msg);
        return Err(UtilsError::InvalidParameter(err_msg));
    } else {
        debug!("Directory already exists: {:?}", dir_path);
    }
    Ok(())
}

/// Ищет файл с указанным именем в директории (нерекурсивно).
///
/// # Arguments
/// * `dir` - Директория для поиска.
/// * `file_name` - Имя файла для поиска.
///
/// # Returns
/// `Ok(Some(PathBuf))`, если файл найден, `Ok(None)`, если не найден.
///
/// # Errors
/// Возвращает `UtilsError::Io`, если есть проблема с чтением директории.
/// Возвращает `UtilsError::InvalidParameter`, если предоставленный `dir` не является директорией.
pub fn find_file_in_dir(dir: &Path, file_name: &str) -> Result<Option<PathBuf>, UtilsError> {
    if !dir.is_dir() {
        return Err(UtilsError::InvalidParameter(format!(
            "Provided path is not a directory: {:?}",
            dir
        )));
    }
    debug!("Searching for file '{}' in directory: {:?}", file_name, dir);
    for entry in fs::read_dir(dir)
        .map_err(|e| UtilsError::io_with_path(e, dir.to_string_lossy().into_owned()))?
    {
        let entry = entry.map_err(|e| UtilsError::io_with_path(e, ".".into()))?;
        let path = entry.path();
        if path.is_file() && path.file_name().map_or(false, |name| name == file_name) {
            info!("Found file: {:?}", path);
            return Ok(Some(path));
        }
    }
    info!("File '{}' not found in directory: {:?}", file_name, dir);
    Ok(None)
}

/// Очищает строку, чтобы ее можно было безопасно использовать как компонент пути.
/// Заменяет потенциально проблемные символы на подчеркивания (`_`).
///
/// # Arguments
/// * `component` - Строка для очистки.
///
/// # Returns
/// Новая `String` с недопустимыми символами, замененными на подчеркивания.
pub fn sanitize_path_component(component: &str) -> String {
    component
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '.' | '_' => c,
            _ => '_',
        })
        .collect()
}
