use std::fs::{self, File};
use std::path::Path;
use tempfile::tempdir;
use utils_crate::error::UtilsError;
use utils_crate::path::{ensure_dir_exists, find_file_in_dir, sanitize_path_component};

#[test]
fn test_ensure_dir_exists_creates_new_pt() -> Result<(), Box<dyn std::error::Error>> {
    let base_dir = tempdir()?;
    let new_dir = base_dir.path().join("test_dir_pt");
    assert!(!new_dir.exists());
    ensure_dir_exists(&new_dir)?;
    assert!(new_dir.exists() && new_dir.is_dir());
    Ok(())
}

#[test]
fn test_ensure_dir_exists_existing_pt() -> Result<(), Box<dyn std::error::Error>> {
    let existing_dir_guard = tempdir()?;
    let existing_dir_path = existing_dir_guard.path();
    ensure_dir_exists(existing_dir_path)?;
    assert!(existing_dir_path.exists() && existing_dir_path.is_dir());
    Ok(())
}

#[test]
fn test_ensure_dir_exists_file_conflict_pt() {
    let base_dir = tempdir().unwrap();
    let file_path = base_dir.path().join("conflict_file_pt");
    File::create(&file_path).unwrap();
    let result = ensure_dir_exists(&file_path);
    assert!(result.is_err());
    match result.unwrap_err() {
        UtilsError::InvalidParameter(msg) => {
            assert!(msg.contains("exists but is not a directory"));
        }
        _ => panic!("Unexpected error type for file conflict"),
    }
}

#[test]
fn test_find_file_in_dir_found_pt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let file_path = dir.path().join("found_me_pt.txt");
    File::create(&file_path)?;
    let found = find_file_in_dir(dir.path(), "found_me_pt.txt")?;
    assert_eq!(found, Some(file_path));
    Ok(())
}

#[test]
fn test_find_file_in_dir_not_found_pt() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let found = find_file_in_dir(dir.path(), "not_found_pt.txt")?;
    assert_eq!(found, None);
    Ok(())
}

#[test]
fn test_find_file_in_dir_target_is_not_dir_pt() {
    let base_dir = tempdir().unwrap();
    let file_as_dir_path = base_dir.path().join("some_file_that_is_not_a_dir.txt");
    File::create(&file_as_dir_path).unwrap();

    let result = find_file_in_dir(&file_as_dir_path, "any_file.txt");
    assert!(result.is_err());
    match result.unwrap_err() {
        UtilsError::InvalidParameter(msg) => {
            assert!(msg.contains("Provided path is not a directory"));
        }
        _ => panic!("Unexpected error type for non-directory input."),
    }
}

#[test]
fn test_sanitize_path_component_valid_pt() {
    assert_eq!(
        sanitize_path_component("valid-name_1.2"),
        "valid-name_1.2"
    );
}

#[test]
fn test_sanitize_path_component_replaces_invalid_pt() {
    assert_eq!(sanitize_path_component("invalid*name?"), "invalid_name_");
    assert_eq!(sanitize_path_component("path/to/file"), "path_to_file");
    assert_eq!(
        sanitize_path_component("Тестовая строка с кириллицей"),
        "_________________________________"
    );
    assert_eq!(sanitize_path_component("a!b@c#d$e%f^g&h*i(j)k"), "a_b_c_d_e_f_g_h_i_j_k");
}

#[test]
fn test_sanitize_path_component_empty_and_all_invalid_pt() {
    assert_eq!(sanitize_path_component(""), "");
    assert_eq!(sanitize_path_component("!@#$%^"), "______");
}
