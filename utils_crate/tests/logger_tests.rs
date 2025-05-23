use std::fs;
use std::io::Read;
use tempfile::tempdir;
use tracing::Level;
use utils_crate::logger::init_tracing_logger;

#[test]
fn test_logger_init_console_only() {
    let _ = init_tracing_logger("test_app_console_only_utils", Level::DEBUG, Level::INFO, None, false);
    tracing::debug!("Console only logger test message from utils_crate.");
}

#[cfg(feature = "logger_utils_feature")]
#[test]
fn test_logger_init_with_file() {
    let temp_dir = tempdir().unwrap();
    let log_file_dir = temp_dir.path();

    let _ = init_tracing_logger(
        "test_app_file_utils",
        Level::INFO,
        Level::DEBUG,
        Some(log_file_dir),
        false,
    );
    
    std::thread::sleep(std::time::Duration::from_millis(300));
    tracing::info!("A message to be logged to file from utils_crate logger_tests.");
    std::thread::sleep(std::time::Duration::from_millis(300));

    let entries: Vec<_> = fs::read_dir(log_file_dir)
        .unwrap()
        .map(|res| res.unwrap().path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .map_or(false, |n| n.to_string_lossy().contains("test_app_file_utils.log"))
        })
        .collect();

    assert!(!entries.is_empty(), "Log file was not created in the directory: {:?}. Files found: {:?}", log_file_dir, fs::read_dir(log_file_dir).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>());

    let mut file_content = String::new();
    let mut found_log_message = false;
    for entry_path in entries {
        if let Ok(mut file) = fs::File::open(&entry_path) {
            file_content.clear();
            if file.read_to_string(&mut file_content).is_ok() {
                if file_content.contains("A message to be logged to file from utils_crate logger_tests.") {
                    found_log_message = true;
                    break;
                }
            }
        }
    }
    assert!(
        found_log_message,
        "Log message not found in any log file. Content checked: '{}'", file_content
    );
}
