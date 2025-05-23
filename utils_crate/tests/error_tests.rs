use std::io;
use utils_crate::error::UtilsError;

#[test]
fn test_io_error_conversion() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
    let path_str = "non_existent_file.txt";
    let utils_err = UtilsError::io_with_path(io_err, path_str.to_string());

    match utils_err {
        UtilsError::Io {
            source,
            path: Some(p),
        } => {
            assert_eq!(source.kind(), io::ErrorKind::NotFound);
            assert_eq!(p, path_str);
            assert!(source.to_string().contains("file not found"));
        }
        _ => panic!("Expected UtilsError::Io variant with path"),
    }

    let io_err_no_path = io::Error::new(io::ErrorKind::Other, "other io error");
    let utils_err_no_path: UtilsError = io_err_no_path.into(); // Используем From<std::io::Error>
    match utils_err_no_path {
        UtilsError::Io { source, path: None } => {
            assert_eq!(source.kind(), io::ErrorKind::Other);
            assert!(source.to_string().contains("other io error"));
        }
        _ => panic!("Expected UtilsError::Io variant without path"),
    }
}

#[test]
fn test_config_error_formatting() {
    let err = UtilsError::Config("Invalid config format".to_string());
    assert_eq!(
        format!("{}", err),
        "Configuration error: Invalid config format"
    );
}

#[test]
fn test_invalid_parameter_formatting() {
    let err = UtilsError::InvalidParameter("Null value not allowed".to_string());
    assert_eq!(
        format!("{}", err),
        "Invalid parameter: Null value not allowed"
    );
}

#[test]
fn test_not_supported_formatting() {
    let err = UtilsError::NotSupported("Legacy API version".to_string());
    assert_eq!(
        format!("{}", err),
        "Operation or feature not supported: Legacy API version"
    );
}

#[test]
fn test_resource_not_found_formatting() {
    let err = UtilsError::ResourceNotFound("Model 'llama-7b' not found".to_string());
    assert_eq!(
        format!("{}", err),
        "Resource not found: Model 'llama-7b' not found"
    );
}


#[test]
fn test_generic_error_formatting() {
    let err = UtilsError::Generic("Something unexpected happened".to_string());
    assert_eq!(
        format!("{}", err),
        "A generic utility error occurred: Something unexpected happened"
    );
}

#[cfg(all(feature = "serde", feature = "serde_json"))]
#[test]
fn test_serde_json_error_conversion_utils() {
    let json_str = "{ \"key\": invalid }"; // Невалидный JSON
    let serde_err_result: Result<serde_json::Value, _> = serde_json::from_str(json_str);
    assert!(serde_err_result.is_err());
    let serde_err = serde_err_result.unwrap_err();

    let utils_err: UtilsError = serde_err.into(); // Используем From<serde_json::Error>

    match utils_err {
        UtilsError::Deserialization(msg) => {
            assert!(msg.contains("JSON error:"), "Error message was: {}", msg);
            // Сообщение об ошибке может немного отличаться в зависимости от версии serde_json
            assert!(
                msg.contains("expected value at line 1 column 10") // serde_json < 1.0.100
                || msg.contains("expected value at line 1 column 11") // serde_json >= 1.0.100
                || msg.contains("key must be a string") // Другой тип ошибки, если бы ключ был невалиден
                || msg.contains("invalid type: identifier `invalid`"), // Более точное сообщение
                "Specific JSON error not found in: {}",
                msg
            );
        }
        _ => panic!(
            "Expected UtilsError::Deserialization variant, got {:?}",
            utils_err
        ),
    }
}

#[cfg(all(feature = "app_config_serde", feature = "toml"))]
#[test]
fn test_toml_de_error_conversion_utils() {
    let toml_str = "key = invalid_toml_value"; // Невалидный TOML
    let toml_err_result: Result<toml::Value, _> = toml::from_str(toml_str);
    assert!(toml_err_result.is_err());
    let toml_err = toml_err_result.unwrap_err();
    let utils_err: UtilsError = toml_err.into(); // Используем From<toml::de::Error>

    match utils_err {
        UtilsError::Deserialization(msg) => {
            assert!(
                msg.contains("TOML deserialization error:"),
                "Error message was: {}",
                msg
            );
            // Сообщение об ошибке может немного отличаться
            assert!(
                msg.contains("invalid_toml_value")
                || msg.contains("invalid type: identifier")
                || msg.contains("expected a value"),
                "Specific TOML error not found in: {}",
                msg
            );
        }
        _ => panic!(
            "Expected UtilsError::Deserialization variant, got {:?}",
            utils_err
        ),
    }
}

#[cfg(all(feature = "app_config_serde", feature = "toml"))]
#[test]
fn test_toml_ser_error_conversion_utils() {
    // Создать ситуацию, когда toml::ser::Error возникает, сложнее,
    // так как обычно ошибки сериализации связаны с типами, не поддерживающими Serialize.
    // Для простого примера, можно представить, что мы пытаемся сериализовать что-то,
    // что toml не поддерживает напрямую (хотя это обычно отлавливается на уровне Serialize).
    // Вместо этого, создадим искусственную ошибку для демонстрации.

    // Этот тест больше для проверки конвертации, чем для реального сценария ошибки сериализации toml.
    // В реальных сценариях, если тип реализует `Serialize`, toml::to_string обычно успешен
    // или ошибка будет другого рода (например, не UTF-8).
    // Для демонстрации `From<toml::ser::Error>` мы можем создать фиктивную ошибку.
    let toml_ser_err = toml::ser::Error::Custom("custom toml serialization error".to_string());
    let utils_err: UtilsError = toml_ser_err.into();

    match utils_err {
        UtilsError::Serialization(msg) => {
            assert!(
                msg.contains("TOML serialization error:"),
                "Error message was: {}",
                msg
            );
            assert!(
                msg.contains("custom toml serialization error"),
                "Specific TOML error not found in: {}",
                msg
            );
        }
        _ => panic!(
            "Expected UtilsError::Serialization variant, got {:?}",
            utils_err
        ),
    }
}
