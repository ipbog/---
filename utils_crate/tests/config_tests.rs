#[cfg(feature = "app_config_serde")]
mod app_config_feature_tests {
    use utils_crate::config::AppConfig;
    use utils_crate::error::UtilsError;
    use tempfile::NamedTempFile;
    use std::io::Write;
    use std::path::Path;

    // Вспомогательные функции для значений по умолчанию, если они не pub в config.rs
    // или используйте AppConfig::default() и проверяйте его поля.
    fn default_models_dir_path_app_test() -> String { "./.model_cache".to_string() }
    fn default_api_host_app_test() -> String { "127.0.0.1".to_string() }
    fn default_api_port_app_test() -> u16 { 8080 }
    fn default_log_level_app_test() -> String { "info".to_string() }
    fn default_theme_app_test() -> String { "dark".to_string() }


    #[test]
    fn test_app_config_default_values() {
        let config = AppConfig::default();
        assert_eq!(config.model_config.model_dir, default_models_dir_path_app_test());
        assert!(config.model_config.preload_models.is_empty());
        assert_eq!(config.model_config.allow_dynamic_loading, true);

        assert_eq!(config.api_config.host, default_api_host_app_test());
        assert_eq!(config.api_config.port, default_api_port_app_test());
        assert_eq!(config.api_config.cors_enabled, false);

        assert_eq!(config.logging_config.level, default_log_level_app_test());
        assert_eq!(config.logging_config.log_file, None);
        assert_eq!(config.logging_config.json_format, false);

        assert_eq!(config.general_settings.theme, default_theme_app_test());
        assert_eq!(config.general_settings.language, "en_US".to_string());
    }

    #[test]
    fn test_app_config_load_from_toml_full() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let toml_content = r#"
            [model_config]
            model_dir = "/custom/models"
            preload_models = ["gemma-7b", "llama-2-13b"]
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
        assert_eq!(config.model_config.model_dir, "/custom/models".to_string());
        assert_eq!(config.model_config.preload_models, vec!["gemma-7b".to_string(), "llama-2-13b".to_string()]);
        assert_eq!(config.model_config.allow_dynamic_loading, false);

        assert_eq!(config.api_config.host, "0.0.0.0".to_string());
        assert_eq!(config.api_config.port, 9090);
        assert_eq!(config.api_config.cors_enabled, true);

        assert_eq!(config.logging_config.level, "debug".to_string());
        assert_eq!(config.logging_config.log_file, Some("/var/log/ai_assistant.log".to_string()));
        assert_eq!(config.logging_config.json_format, true);

        assert_eq!(config.general_settings.theme, "light".to_string());
        assert_eq!(config.general_settings.language, "ru_RU".to_string());
    }

    #[test]
    fn test_app_config_load_from_toml_partial() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let toml_content = r#"
            [api_config]
            port = 9999

            [logging_config]
            level = "trace"
        "#;
        writeln!(temp_file, "{}", toml_content).unwrap();
        let config = AppConfig::load_from_toml(temp_file.path()).unwrap();

        // Проверяем измененные значения
        assert_eq!(config.api_config.port, 9999);
        assert_eq!(config.logging_config.level, "trace".to_string());

        // Проверяем, что остальные значения взяты по умолчанию
        assert_eq!(config.api_config.host, default_api_host_app_test());
        assert_eq!(config.model_config.model_dir, default_models_dir_path_app_test());
        assert_eq!(config.general_settings.theme, default_theme_app_test());
    }

    #[test]
    fn test_app_config_load_from_toml_file_not_found() {
        let non_existent_path = Path::new("/абсолютно/несуществующий/путь/config.toml");
        let config = AppConfig::load_from_toml(non_existent_path).unwrap();
        // Ожидаем, что вернется конфигурация по умолчанию
        assert_eq!(config, AppConfig::default());
    }

    #[test]
    fn test_app_config_load_from_toml_invalid_toml_syntax() {
        let mut temp_file = NamedTempFile::new().unwrap();
        // Невалидный TOML: отсутствует закрывающая кавычка
        let invalid_toml_content = r#"
            [api_config]
            host = "0.0.0.0
        "#;
        writeln!(temp_file, "{}", invalid_toml_content).unwrap();

        let result = AppConfig::load_from_toml(temp_file.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            UtilsError::Deserialization(msg) => {
                assert!(msg.contains("Ошибка десериализации TOML:"));
            }
            other_err => panic!("Ожидалась ошибка UtilsError::Deserialization, получено {:?}", other_err),
        }
    }

     #[test]
    fn test_app_config_load_from_toml_type_mismatch() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let invalid_toml_content = r#"
            [api_config]
            port = "не_число" 
        "#; // port должен быть u16, а не строкой
        writeln!(temp_file, "{}", invalid_toml_content).unwrap();

        let result = AppConfig::load_from_toml(temp_file.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            UtilsError::Deserialization(msg) => {
                 assert!(msg.contains("Ошибка десериализации TOML:"));
                 // Сообщение от toml может варьироваться, но должно указывать на проблему с типом
                 assert!(msg.contains("port") && (msg.contains("invalid type: string") || msg.contains("expected an integer")));
            }
            other_err => panic!("Ожидалась ошибка UtilsError::Deserialization, получено {:?}", other_err),
        }
    }
}
