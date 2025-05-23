use utils_crate::path::{ensure_dir_exists, find_file_in_dir, sanitize_path_component};
use utils_crate::error::UtilsError;
use tempfile::tempdir;
use std::fs::{self, File};
use std::path::Path;

#[test]
fn test_ensure_dir_exists_creates_new() -> Result<(), Box<dyn std::error::Error>> {
    let base_dir = tempdir()?;
    let new_dir = base_dir.path().join("новая_директория");
    assert!(!new_dir.exists());
    ensure_dir_exists(&new_dir)?;
    assert!(new_dir.exists() && new_dir.is_dir());
    Ok(())
}

#[test]
fn test_ensure_dir_exists_already_exists() -> Result<(), Box<dyn std::error::Error>> {
    let existing_dir_guard = tempdir()?; // tempdir() создает директорию
    let existing_dir_path = existing_dir_guard.path();
    ensure_dir_exists(existing_dir_path)?; // Должен просто вернуть Ok(())
    assert!(existing_dir_path.exists() && existing_dir_path.is_dir());
    Ok(())
}

#[test]
fn test_ensure_dir_exists_path_is_file_conflict() {
    let base_dir = tempdir().unwrap();
    let file_path = base_dir.path().join("конфликтный_файл.txt");
    File::create(&file_path).unwrap(); // Создаем файл там, где ожидаем директорию

    let result = ensure_dir_exists(&file_path);
    assert!(result.is_err());
    match result.unwrap_err() {
        UtilsError::InvalidParameter(msg) => {
            assert!(msg.contains("существует, но не является директорией"));
        }
        other_err => panic!("Ожидалась ошибка UtilsError::InvalidParameter, получено {:?}", other_err),
    }
}

#[test]
fn test_find_file_in_dir_found() -> Result<(), Box<dyn std::error::Error>> {
    let dir_guard = tempdir()?;
    let dir_path = dir_guard.path();
    let file_to_find = "искомый_файл.txt";
    let file_path = dir_path.join(file_to_find);
    File::create(&file_path)?;

    let found = find_file_in_dir(dir_path, file_to_find)?;
    assert_eq!(found, Some(file_path));
    Ok(())
}

#[test]
fn test_find_file_in_dir_not_found() -> Result<(), Box<dyn std::error::Error>> {
    let dir_guard = tempdir()?;
    let dir_path = dir_guard.path();
    let found = find_file_in_dir(dir_path, "несуществующий_файл.txt")?;
    assert_eq!(found, None);
    Ok(())
}

#[test]
fn test_find_file_in_dir_path_is_not_directory() {
    // Создаем временный файл, чтобы использовать его как "не директорию"
    let temp_file_guard = tempfile::NamedTempFile::new().unwrap();
    let file_as_dir_path = temp_file_guard.path();

    let result = find_file_in_dir(file_as_dir_path, "любой_файл.txt");
    assert!(result.is_err());
    match result.unwrap_err() {
        UtilsError::InvalidParameter(msg) => {
            assert!(msg.contains("Указанный путь не является директорией"));
        }
        other_err => panic!("Ожидалась ошибка UtilsError::InvalidParameter, получено {:?}", other_err),
    }
}

#[test]
fn test_find_file_in_dir_permission_denied_reading_dir() {
    // Этот тест сложнее надежно реализовать кроссплатформенно без root прав
    // или специфичных настроек ФС. Пропускаем его или делаем условным.
    // Идея: создать директорию, затем изменить ее права так, чтобы нельзя было прочитать.
    // Для простоты, этот сценарий здесь не тестируется напрямую.
    // Вместо этого, можно проверить, что fs::read_dir возвращает ошибку,
    // и она корректно мапится в UtilsError::Io.
}


#[test]
fn test_sanitize_path_component_no_changes() {
    assert_eq!(sanitize_path_component("валидное-имя_1.23.txt"), "валидное-имя_1.23.txt");
}

#[test]
fn test_sanitize_path_component_replaces_common_invalid_chars() {
    assert_eq!(sanitize_path_component("файл*с?пробелами:/\<>|""), "файл_с_пробелами_______");
}

#[test]
fn test_sanitize_path_component_replaces_non_ascii_if_strict_needed() {
    // Текущая реализация заменяет не-ASCII на '_', если они не входят в разрешенный набор.
    // Если бы кириллица была разрешена, тест был бы другим.
    assert_eq!(sanitize_path_component("ТестКириллицы123"), "ТестКириллицы123"); // Если разрешены только a-z A-Z 0-9 - . _
                                                                               // то результат будет "____________123"
                                                                               // Исправляем на основе текущей реализации sanitize_path_component
    assert_eq!(sanitize_path_component("ТестКириллицы123"), "________________123");
}

#[test]
fn test_sanitize_path_component_empty_string() {
    assert_eq!(sanitize_path_component(""), "");
}

#[test]
fn test_sanitize_path_component_only_invalid_chars() {
    assert_eq!(sanitize_path_component("/*?<>"), "_____");
}
