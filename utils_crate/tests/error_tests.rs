use utils_crate::error::UtilsError;
use std::io;

#[test]
fn test_io_error_conversion_with_path() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "файл не найден");
    let path_str = "несуществующий_файл.txt";
    let utils_err = UtilsError::io_with_path(io_err, path_str.to_string());

    match utils_err {
        UtilsError::Io { source, path } => {
            assert_eq!(source.kind(), io::ErrorKind::NotFound);
            assert_eq!(path, Some(path_str.to_string()));
            assert_eq!(format!("{}", source), "файл не найден");
        }
        _ => panic!("Ожидался вариант UtilsError::Io с путем"),
    }
}

#[test]
fn test_io_error_conversion_from_std() {
    let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "доступ запрещен");
    let utils_err: UtilsError = io_err.into(); // Используем #[from]

    match utils_err {
        UtilsError::Io { source, path } => {
            assert_eq!(source.kind(), io::ErrorKind::PermissionDenied);
            assert!(path.is_none()); // #[from] не добавляет путь автоматически
            assert_eq!(format!("{}", source), "доступ запрещен");
        }
        _ => panic!("Ожидался вариант UtilsError::Io без пути"),
    }
}


#[test]
fn test_error_display_formatting() {
    assert_eq!(
        format!("{}", UtilsError::Config("Неверный формат".to_string())),
        "Ошибка конфигурации: Неверный формат"
    );
    assert_eq!(
        format!("{}", UtilsError::InvalidParameter("Нулевое значение".to_string())),
        "Неверный параметр: Нулевое значение"
    );
    assert_eq!(
        format!("{}", UtilsError::NotSupported("Эта фича".to_string())),
        "Операция или функция не поддерживается: Эта фича"
    );
    assert_eq!(
        format!("{}", UtilsError::ResourceNotFound("Модель X".to_string())),
        "Ресурс не найден: Модель X"
    );
    assert_eq!(
        format!("{}", UtilsError::Generic("Что-то пошло не так".to_string())),
        "Произошла общая ошибка утилиты: Что-то пошло не так"
    );
}

#[cfg(all(feature = "serde", feature = "serde_json"))]
mod serde_json_errors {
    use super::*; // Для UtilsError
    use serde_json;

    #[test]
    fn test_serde_json_deserialization_error_conversion() {
        let json_str = "{ \"ключ\": "; // Невалидный JSON (неожиданный конец)
        let serde_err_result: Result<serde_json::Value, _> = serde_json::from_str(json_str);
        assert!(serde_err_result.is_err());
        let serde_err = serde_err_result.unwrap_err();

        let utils_err: UtilsError = serde_err.into();

        match utils_err {
            UtilsError::Deserialization(msg) => {
                assert!(msg.contains("Ошибка JSON (десериализация):"));
                assert!(msg.contains("EOF while parsing a value")); // Специфичное сообщение от serde_json
            }
            _ => panic!("Ожидался UtilsError::Deserialization, получено {:?}", utils_err),
        }
    }
}


#[cfg(all(feature = "app_config_serde", feature = "toml"))]
mod toml_errors {
    use super::*; // Для UtilsError
    use toml;

    #[test]
    fn test_toml_deserialization_error_conversion() {
        let toml_str = "ключ = [1, 2,"; // Невалидный TOML (незакрытый массив)
        let toml_err_result: Result<toml::Value, _> = toml::from_str(toml_str);
        assert!(toml_err_result.is_err());
        let toml_err = toml_err_result.unwrap_err();
        let utils_err: UtilsError = toml_err.into();

        match utils_err {
            UtilsError::Deserialization(msg) => {
                assert!(msg.contains("Ошибка десериализации TOML:"));
                // Сообщение об ошибке от toml может немного отличаться
                assert!(msg.contains("expected `]`"));
            }
            _ => panic!("Ожидался UtilsError::Deserialization, получено {:?}", utils_err),
        }
    }

    #[test]
    fn test_toml_serialization_error_conversion() {
        // Создать ситуацию, когда toml::ser::Error возникает, сложнее,
        // так как он обычно связан с несериализуемыми типами,
        // которые не должны появляться в AppConfig.
        // Этот тест больше для демонстрации конвертации.
        // Предположим, у нас есть ошибка сериализации (гипотетическая).
        struct NonSerializable; // Не реализует Serialize
        // let toml_ser_err = toml::to_string(&NonSerializable).unwrap_err(); // Это не скомпилируется
        // Вместо этого создадим ошибку вручную для теста конвертации
        let dummy_toml_ser_error_string = "ошибка сериализации TOML (пример)";
        let toml_ser_err = toml::ser::Error::custom(dummy_toml_ser_error_string.to_string());


        let utils_err: UtilsError = toml_ser_err.into();
        match utils_err {
            UtilsError::Serialization(msg) => {
                assert!(msg.contains("Ошибка сериализации TOML:"));
                assert!(msg.contains(dummy_toml_ser_error_string));
            }
            _ => panic!("Ожидался UtilsError::Serialization, получено {:?}", utils_err),
        }
    }
}
