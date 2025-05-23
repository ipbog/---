use std::fs;
use std::io::Read;
use tempfile::tempdir;
use tracing::Level;
use utils_crate::logger::init_tracing_logger;

#[test]
fn test_logger_init_console_only() {
    // Инициализация логгера может быть глобальной, поэтому тесты, изменяющие его,
    // лучше запускать последовательно или убедиться, что они не влияют друг на друга.
    // tracing_subscriber::fmt::try_init() вернет ошибку, если логгер уже инициализирован.
    // Для тестов можно использовать `set_default` с `dispatcher`, но это усложнит.
    // Простой вариант - игнорировать ошибку инициализации, если она уже произошла.
    let _ = init_tracing_logger("test_app_console_only_utils", Level::DEBUG, Level::INFO, None, false);
    tracing::debug!("Console only logger test message from utils_crate logger_tests (debug).");
    tracing::info!("Console only logger test message from utils_crate logger_tests (info).");
    // Проверить вывод в консоль вручную или через capture, если тестовый фреймворк поддерживает.
}

#[cfg(feature = "logger_utils_feature")]
#[test]
fn test_logger_init_with_file() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let log_file_dir = temp_dir.path();
    let app_name = "test_app_file_utils";

    // Убедимся, что логгер инициализируется для этого теста.
    // Если он уже был инициализирован глобально, это может не сработать как ожидается
    // без более сложной настройки управления глобальным подписчиком.
    // Для простоты, предполагаем, что этот тест может выполняться в среде, где логгер еще не установлен
    // или его можно переинициализировать (хотя try_init предотвращает это).
    // Лучше всего, если тесты, затрагивающие глобальное состояние, изолированы.
    match init_tracing_logger(
        app_name,
        Level::INFO,  // Console level
        Level::DEBUG, // File level
        Some(log_file_dir),
        false,
    ) {
        Ok(()) => tracing::info!("Logger for test_logger_init_with_file initialized."),
        Err(e) => {
            // Если логгер уже инициализирован, мы можем продолжить, но вывод может идти не туда.
            // Для CI это может быть проблемой.
            eprintln!("Could not initialize logger for test: {:?}. This might affect test results if logger was already set globally.", e);
            // Все равно попробуем записать лог, возможно, он попадет в ранее настроенный файл.
        }
    }


    let log_message_debug = "DEBUG: A message to be logged to file from utils_crate logger_tests.";
    let log_message_info = "INFO: Another message, also to file from utils_crate logger_tests.";

    tracing::debug!("{}", log_message_debug); // Этот должен пойти в файл (DEBUG >= file_level DEBUG)
    tracing::info!("{}", log_message_info);  // Этот должен пойти и в консоль, и в файл (INFO >= console_level INFO, INFO >= file_level DEBUG)

    // Даем время на запись в файл
    std::thread::sleep(std::time::Duration::from_millis(500));

    let entries: Vec<_> = fs::read_dir(log_file_dir)
        .unwrap_or_else(|e| panic!("Failed to read log directory {:?}: {}", log_file_dir, e))
        .map(|res| res.unwrap().path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .map_or(false, |n| {
                        let s = n.to_string_lossy();
                        s.starts_with(app_name) && s.ends_with(".log")
                    })
        })
        .collect();

    assert!(!entries.is_empty(), "Log file was not created in the directory: {:?}. Files found: {:?}", log_file_dir, fs::read_dir(log_file_dir).unwrap().map(|e| e.unwrap().path()).collect::<Vec<_>>());

    let mut file_content = String::new();
    let mut found_debug_message = false;
    let mut found_info_message = false;

    // Обычно создается один файл лога за день (или при запуске, если имя не включает дату)
    // Например, test_app_file_utils.log.YYYY-MM-DD
    // Мы ищем файл, который начинается с app_name и заканчивается на .log
    // tracing_appender::rolling::daily создает файлы вида `prefix.ГГГГ-ММ-ДД`
    // Если имя файла `app_name.log`, то будет `app_name.log.ГГГГ-ММ-ДД`
    // Если имя файла `app_name`, то будет `app_name.ГГГГ-ММ-ДД`
    // В нашем случае `format!("{}.log", app_name)` -> `test_app_file_utils.log.YYYY-MM-DD`

    for entry_path in &entries {
        if let Ok(mut file) = fs::File::open(entry_path) {
            file_content.clear(); // Очищаем для каждого файла
            if file.read_to_string(&mut file_content).is_ok() {
                if file_content.contains(log_message_debug) {
                    found_debug_message = true;
                }
                if file_content.contains(log_message_info) {
                    found_info_message = true;
                }
                if found_debug_message && found_info_message {
                    break;
                }
            }
        } else {
            eprintln!("Could not open log file: {:?}", entry_path);
        }
    }
    assert!(
        found_debug_message,
        "DEBUG log message not found in any log file. Searched files in {:?}. Last content checked: '{}'", entries, file_content
    );
    assert!(
        found_info_message,
        "INFO log message not found in any log file. Searched files in {:?}. Last content checked: '{}'", entries, file_content
    );
}
