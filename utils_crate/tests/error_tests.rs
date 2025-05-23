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
        }
        _ => panic!("Expected UtilsError::Io variant with path"),
    }

    let io_err_no_path = io::Error::new(io::ErrorKind::Other, "other io error");
    let utils_err_no_path: UtilsError = io_err_no_path.into();
    match utils_err_no_path {
        UtilsError::Io { source, path: None } => {
            assert_eq!(source.kind(), io::ErrorKind::Other);
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
    let json_str = "{ \"key\": invalid }";
    let serde_err_result: Result<serde_json::Value, _> = serde_json::from_str(json_str);
    assert!(serde_err_result.is_err());
    let serde_err = serde_err_result.unwrap_err();

    let utils_err: UtilsError = serde_err.into();

    match utils_err {
        UtilsError::Deserialization(msg) => {
            assert!(msg.contains("JSON error:"), "Error message was: {}", msg);
            assert!(
                msg.contains("expected value at line 1 column 10")
                    || msg.contains("key must be a string"),
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
    let toml_str = "key = invalid_toml_value";
    let toml_err_result: Result<toml::Value, _> = toml::from_str(toml_str);
    assert!(toml_err_result.is_err());
    let toml_err = toml_err_result.unwrap_err();
    let utils_err: UtilsError = toml_err.into();

    match utils_err {
        UtilsError::Deserialization(msg) => {
            assert!(
                msg.contains("TOML deserialization error:"),
                "Error message was: {}",
                msg
            );
            assert!(
                msg.contains("invalid_toml_value") || msg.contains("invalid type: identifier")
            );
        }
        _ => panic!(
            "Expected UtilsError::Deserialization variant, got {:?}",
            utils_err
        ),
    }
}
