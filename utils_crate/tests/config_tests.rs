#[cfg(feature = "app_config_serde")]
mod app_config_feature_tests {
    use std::io::Write;
    use std::path::Path;
    use tempfile::NamedTempFile;
    use utils_crate::config::AppConfig;
    use utils_crate::error::UtilsError;

    fn default_models_dir_path_app_test() -> String {
        "./.model_cache".to_string()
    }
    fn default_api_host_app_test() -> String {
        "127.0.0.1".to_string()
    }

    #[test]
    fn test_app_config_default_values_ct() {
        let config = AppConfig::default();
        assert_eq!(
            config.model_config.model_dir,
            default_models_dir_path_app_test()
        );
        assert_eq!(config.api_config.port, 8080);
        assert_eq!(config.logging_config.level, "info".to_string());
        assert_eq!(config.general_settings.theme, "dark".to_string());
    }

    #[test]
    fn test_app_config_load_from_toml_exists_ct() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let toml_content = r#"
            [model_config]
            model_dir = "/custom/models"
            preload_models = ["gemma-7b"]
            allow_dynamic_loading = false

            [api_config]
            host = "0.0.0.0"
            port = 9090
            cors_enabled = true

            [logging_config]
            level = "debug"
            log_file = "/var/log/ai_assistant.log"
            json_format = true

            [general_settings]
            theme = "light"
            language = "ru_RU"
        "#;
        writeln!(temp_file, "{}", toml_content).unwrap();

        let config = AppConfig::load_from_toml(temp_file.path()).unwrap();
        assert_eq!(
            config.model_config.model_dir,
            "/custom/models".to_string()
        );
        assert_eq!(
            config.model_config.preload_models,
            vec!["gemma-7b".to_string()]
        );
        assert!(!config.model_config.allow_dynamic_loading);
        assert_eq!(config.api_config.host, "0.0.0.0".to_string());
        assert_eq!(config.api_config.port, 9090);
        assert!(config.api_config.cors_enabled);
        assert_eq!(config.logging_config.level, "debug".to_string());
        assert_eq!(
            config.logging_config.log_file,
            Some("/var/log/ai_assistant.log".to_string())
        );
        assert!(config.logging_config.json_format);
        assert_eq!(config.general_settings.theme, "light".to_string());
        assert_eq!(config.general_settings.language, "ru_RU".to_string());
    }

    #[test]
    fn test_app_config_partial_deserialization_ct() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let toml_content = r#"
            [api_config]
            port = 9999
        "#;
        writeln!(temp_file, "{}", toml_content).unwrap();
        let config = AppConfig::load_from_toml(temp_file.path()).unwrap();

        // Проверяем измененное значение
        assert_eq!(config.api_config.port, 9999);
        // Проверяем, что остальные значения из этой секции остались по умолчанию
        assert_eq!(config.api_config.host, default_api_host_app_test());
        assert!(!config.api_config.cors_enabled); // default is false

        // Проверяем, что значения из других секций остались по умолчанию
        assert_eq!(
            config.model_config.model_dir,
            default_models_dir_path_app_test()
        );
        assert_eq!(config.logging_config.level, "info".to_string());
    }

    #[test]
    fn test_app_config_file_not_found_ct() {
        let non_existent_path = Path::new("/totally/non/existent/path/config.toml");
        let config = AppConfig::load_from_toml(non_existent_path).unwrap();
        // Должна вернуться конфигурация по умолчанию
        assert_eq!(config, AppConfig::default());
    }

    #[test]
    fn test_app_config_invalid_toml_ct() {
        let mut temp_file = NamedTempFile::new().unwrap();
        // Невалидный TOML: port должен быть числом, а не строкой
        let invalid_toml_content = r#"
            [api_config]
            port = "not_a_number"
        "#;
        writeln!(temp_file, "{}", invalid_toml_content).unwrap();

        let result = AppConfig::load_from_toml(temp_file.path());
        assert!(result.is_err());
        if let Err(UtilsError::Config(msg)) = result {
            assert!(msg.contains("Failed to parse AppConfig from TOML"));
            assert!(msg.contains("invalid type: string "not_a_number", expected integer"));
        } else {
            panic!(
                "Expected a Config error for invalid TOML, got {:?}",
                result
            );
        }
    }

    #[test]
    fn test_app_config_empty_file_ct() {
        let mut temp_file = NamedTempFile::new().unwrap();
        // Пустой файл
        let empty_toml_content = "";
        writeln!(temp_file, "{}", empty_toml_content).unwrap();

        let config = AppConfig::load_from_toml(temp_file.path()).unwrap();
        // Должна вернуться конфигурация по умолчанию, так как все поля будут десериализованы
        // с использованием #[serde(default = ...)]
        assert_eq!(config, AppConfig::default());
    }

    #[test]
    fn test_app_config_unknown_fields_ct() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let toml_content_with_unknown = r#"
            [api_config]
            port = 1234
            unknown_field_api = "some_value"

            [new_unknown_section]
            key = "value"
        "#;
        // По умолчанию serde(deny_unknown_fields) не используется, поэтому неизвестные поля должны игнорироваться
        writeln!(temp_file, "{}", toml_content_with_unknown).unwrap();
        let config = AppConfig::load_from_toml(temp_file.path()).unwrap();

        assert_eq!(config.api_config.port, 1234);
        // Проверяем, что остальные поля остались по умолчанию
        assert_eq!(config.api_config.host, default_api_host_app_test());
        assert_eq!(
            config.model_config.model_dir,
            default_models_dir_path_app_test()
        );
    }
}
