use utils_crate::logger::init_tracing_logger;
use utils_crate::error::UtilsError; // Для проверки типа ошибки, если нужно
use tracing::Level;
use tempfile::tempdir;
use std::io::Read;
use std::fs;
use std::path::Path;

// Вспомогательная функция для проверки содержимого файла
fn check_log_file_content(log_dir: &Path, app_name_for_file: &str, expected_message: &str) -> bool {
    // Даем немного времени на запись в файл, особенно если тесты быстрые
    std::thread::sleep(std::time::Duration::from_millis(100));

    let entries: Vec<_> = fs::read_dir(log_dir)
        .expect("Не удалось прочитать директорию логов")
        .filter_map(Result::ok)
        .map(|res| res.path())
        .filter(|p| p.is_file() && p.file_name()
            .map_or(false, |n| n.to_string_lossy().starts_with(app_name_for_file) && n.to_string_lossy().ends_with(".log"))
        )
        .collect();

    if entries.is_empty() {
        println!("Лог-файл для {} не найден в {:?}", app_name_for_file, log_dir);
        return false;
    }

    let mut file_content = String::new();
    for entry_path in entries { // Проверяем все файлы, так как ротация может создать несколько
        if let Ok(mut file) = fs::File::open(&entry_path) {
            file_content.clear(); // Очищаем перед чтением нового файла
            if file.read_to_string(&mut file_content).is_ok() {
                if file_content.contains(expected_message) {
                    return true; // Сообщение найдено
                }
            } else {
                 println!("Не удалось прочитать содержимое файла: {:?}", entry_path);
            }
        } else {
            println!("Не удалось открыть файл: {:?}", entry_path);
        }
    }
    println!("Ожидаемое сообщение '{}' не найдено ни в одном лог-файле в {:?}", expected_message, log_dir);
    false
}


// Тесты инициализации глобального логгера могут быть сложными из-за глобального состояния.
// `serial_test` может помочь, если тесты влияют друг на друга.
// `cargo add serial_test --dev`
// use serial_test::serial;

#[test]
// #[serial] // Раскомментируйте, если тесты конфликтуют
fn test_logger_init_console_only() {
    // Этот тест может конфликтовать с другими, если они тоже инициализируют глобальный логгер.
    // В реальном приложении логгер инициализируется один раз.
    // Попытка инициализировать логгер, если он уже инициализирован, вернет ошибку.
    // Мы можем игнорировать эту ошибку в тесте, если она связана с повторной инициализацией.
    match init_tracing_logger("тест_консоль", Level::DEBUG, Level::INFO, None, false) {
        Ok(()) => {
            tracing::info!("Тест консольного логгера: INFO сообщение.");
            tracing::debug!("Тест консольного логгера: DEBUG сообщение.");
            // Визуально проверить вывод в консоль во время теста
        }
        Err(UtilsError::Generic(msg)) if msg.contains("Failed to initialize logger") || msg.contains("once") => {
            // Логгер уже был инициализирован другим тестом, это нормально для тестового окружения.
            // Можно вывести предупреждение, но тест не должен падать.
            println!("[ПРЕДУПРЕЖДЕНИЕ] Логгер уже инициализирован: {}", msg);
        }
        Err(e) => panic!("Неожиданная ошибка при инициализации логгера: {:?}", e),
    }
}

#[cfg(feature = "logger_utils_feature")]
#[test]
// #[serial] // Раскомментируйте, если тесты конфликтуют
fn test_logger_init_with_file() {
    let temp_dir = tempdir().unwrap();
    let log_file_dir = temp_dir.path();
    let app_name = "тест_файл_лог";
    let log_message = "Сообщение для записи в файл из utils_crate (тест).";

    match init_tracing_logger(app_name, Level::INFO, Level::DEBUG, Some(log_file_dir), false) {
        Ok(()) => {
            tracing::info!("{}", log_message); // INFO должно попасть в файл, если file_level DEBUG
            tracing::debug!("Отладочное сообщение в файл."); // DEBUG должно попасть в файл
        }
        Err(UtilsError::Generic(msg)) if msg.contains("Failed to initialize logger") || msg.contains("once") => {
            println!("[ПРЕДУПРЕЖДЕНИЕ] Логгер уже инициализирован (файл): {}. Попытка записать сообщение.", msg);
            // Если логгер уже был инициализирован, он может быть настроен по-другому.
            // Этот тест может быть неточным в таком случае.
            // Для надежности лучше использовать #[serial] или убедиться, что этот тест запускается первым.
            tracing::info!("{}", log_message);
            tracing::debug!("Отладочное сообщение в файл (повторная инициализация).");
        }
        Err(e) => panic!("Неожиданная ошибка при инициализации логгера (файл): {:?}", e),
    }

    assert!(check_log_file_content(log_file_dir, app_name, log_message), "Сообщение INFO не найдено в лог-файле.");
    assert!(check_log_file_content(log_file_dir, app_name, "Отладочное сообщение в файл"), "Сообщение DEBUG не найдено в лог-файле.");
}

#[cfg(feature = "logger_utils_feature")]
#[test]
// #[serial]
fn test_logger_file_creation_failure_does_not_panic() {
    // Используем путь, куда мы не можем писать (например, корень файловой системы без прав)
    // Это может потребовать специфичных настроек для CI/CD.
    // Для локального теста можно создать read-only директорию.
    // Здесь мы просто используем несуществующий путь в надежде, что create_dir_all не сработает,
    // но это не гарантирует ошибку прав доступа.
    let non_writable_dir = Path::new("/non_existent_root_dir_for_log_test/logs"); 

    // Ожидаем, что init_tracing_logger не запаникует, а выведет предупреждение в stderr
    // и вернет Ok(()), так как консольный логгер должен работать.
    match init_tracing_logger("тест_ошибка_файла", Level::INFO, Level::DEBUG, Some(non_writable_dir), false) {
         Ok(()) => {
            // Успех, если не было паники. Предупреждение должно было быть выведено в stderr.
            tracing::info!("Логгер инициализирован, несмотря на ошибку создания директории для файла.");
        }
        Err(UtilsError::Generic(msg)) if msg.contains("Failed to initialize logger") || msg.contains("once") => {
             println!("[ПРЕДУПРЕЖДЕНИЕ] Логгер уже инициализирован (ошибка файла): {}", msg);
        }
        Err(e) => panic!("Неожиданная ошибка при инициализации логгера (ошибка файла): {:?}", e),
    }
    // Проверить stderr вручную или с помощью capture не так просто в стандартных тестах.
}
