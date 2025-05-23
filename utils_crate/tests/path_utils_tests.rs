use std::fs::{self, File};
use std::io;
use std::path::Path;
use tempfile::tempdir;
use utils_crate::error::UtilsError;
use utils_crate::path_utils::{ensure_dir_exists, find_file_in_dir, sanitize_path_component}; // ИЗМЕНЕНО

#[test]
fn test_ensure_dir_exists_creates_new_put() -> Result<(), Box<dyn std::error::Error>> {
    let base_dir = tempdir()?;
    let new_dir = base_dir.path().join("test_dir_put");
    assert!(!new_dir.exists());
    ensure_dir_exists(&new_dir)?;
    assert!(new_dir.exists() && new_dir.is_dir());
    Ok(())
}

#[test]
fn test_ensure_dir_exists_existing_put() -> Result<(), Box<dyn std::error::Error>> {
    let existing_dir_guard = tempdir()?;
    let existing_dir_path = existing_dir_guard.path();
    // Убедимся, что директория действительно существует перед вызовом
    assert!(existing_dir_path.exists() && existing_dir_path.is_dir());
    ensure_dir_exists(existing_dir_path)?; // Должен просто вернуть Ok(())
    assert!(existing_dir_path.exists() && existing_dir_path.is_dir()); // Проверка для уверенности
    Ok(())
}

#[test]
fn test_ensure_dir_exists_file_conflict_put() {
    let base_dir = tempdir().unwrap();
    let file_path = base_dir.path().join("conflict_file_put");
    File::create(&file_path).unwrap(); // Создаем файл там, где ожидаем директорию
    assert!(file_path.exists() && file_path.is_file());

    let result = ensure_dir_exists(&file_path);
    assert!(result.is_err());
    match result.unwrap_err() {
        UtilsError::InvalidParameter(msg) => {
            assert!(msg.contains("exists but is not a directory"));
            assert!(msg.contains("conflict_file_put"));
        }
        other_err => panic!("Unexpected error type for file conflict: {:?}", other_err),
    }
}

// Тест для случая, когда создание директории не удается из-за прав доступа (сложно симулировать кроссплатформенно)
// Можно было бы создать read-only директорию и попытаться создать в ней поддиректорию.

#[test]
fn test_find_file_in_dir_found_put() -> Result<(), Box<dyn std::error::Error>> {
    let dir_guard = tempdir()?;
    let dir = dir_guard.path();
    let file_path = dir.join("found_me_put.txt");
    File::create(&file_path)?;

    let found = find_file_in_dir(dir, "found_me_put.txt")?;
    assert_eq!(found, Some(file_path));
    Ok(())
}

#[test]
fn test_find_file_in_dir_not_found_put() -> Result<(), Box<dyn std::error::Error>> {
    let dir_guard = tempdir()?;
    let dir = dir_guard.path();
    // Создадим другой файл, чтобы директория не была пустой
    File::create(dir.join("another_file.txt"))?;

    let found = find_file_in_dir(dir, "not_found_put.txt")?;
    assert_eq!(found, None);
    Ok(())
}

#[test]
fn test_find_file_in_dir_empty_dir_put() -> Result<(), Box<dyn std::error::Error>> {
    let dir_guard = tempdir()?;
    let dir = dir_guard.path();
    let found = find_file_in_dir(dir, "any_file.txt")?;
    assert_eq!(found, None);
    Ok(())
}


#[test]
fn test_find_file_in_dir_target_is_not_dir_put() {
    let base_dir = tempdir().unwrap();
    let file_as_dir_path = base_dir.path().join("some_file_that_is_not_a_dir.txt");
    File::create(&file_as_dir_path).unwrap();

    let result = find_file_in_dir(&file_as_dir_path, "any_file.txt");
    assert!(result.is_err());
    match result.unwrap_err() {
        UtilsError::InvalidParameter(msg) => {
            assert!(msg.contains("Provided path is not a directory"));
            assert!(msg.contains("some_file_that_is_not_a_dir.txt"));
        }
        other_err => panic!("Unexpected error type for non-directory input: {:?}", other_err),
    }
}

// Тест для find_file_in_dir с ошибкой чтения директории (сложно симулировать)

#[test]
fn test_sanitize_path_component_valid_put() {
    assert_eq!(
        sanitize_path_component("valid-name_1.2"),
        "valid-name_1.2"
    );
    assert_eq!(sanitize_path_component("abc_XYZ-123.dot"), "abc_XYZ-123.dot");
}

#[test]
fn test_sanitize_path_component_replaces_invalid_put() {
    assert_eq!(sanitize_path_component("invalidname?"), "invalidname_");
    assert_eq!(sanitize_path_component("path/to/file"), "path_to_file");
    assert_eq!(
        sanitize_path_component("Тестовая строка с кириллицей"),
        "_________________________________" // Каждый кириллический символ заменяется на '_'
    );
    assert_eq!(sanitize_path_component("a!b@c#d$e%f^g&h*i(j)k"), "a_b_c_d_e_f_g_h_i_j_k");
    assert_eq!(sanitize_path_component(" leading_space"), "_leading_space");
    assert_eq!(sanitize_path_component("trailing_space "), "trailing_space_");
    assert_eq!(sanitize_path_component("new\nline"), "new_line");
    assert_eq!(sanitize_path_component("tab\tchar"), "tab_char");
}

#[test]
fn test_sanitize_path_component_empty_and_all_invalid_put() {
    assert_eq!(sanitize_path_component(""), "");
    assert_eq!(sanitize_path_component("!@#$%^"), "______");
    assert_eq!(sanitize_path_component("   "), "___");
}

#[test]
fn test_sanitize_path_component_with_underscore_put() {
    assert_eq!(sanitize_path_component("name_with_underscore"), "name_with_underscore");
    assert_eq!(sanitize_path_component("name__with__double"), "name__with__double");
}
