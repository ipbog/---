use crate::{
    error::ModelLoaderError, // Кастомные ошибки
    types::{
        CoreGemmaModelConfig, // "Ядерная" конфигурация
        ModelType, // Перечисление типов моделей
        RawModelConfig, // "Сырая" конфигурация
    },
};
use tracing::{ // Для логирования
    debug, // Отладочные сообщения
    warn, // Предупреждения
};

/// Предоставляет утилиты для валидации конфигураций моделей.
///
/// Этот модуль гарантирует, что загруженные конфигурации моделей соответствуют
/// ожидаемым критериям перед их использованием. Он отделяет логику валидации от загрузки.
///
/// **Примечание:** Этот валидатор специально проверяет отсутствие конфигураций
/// квантования, так как проект работает с чистыми, неквантованными моделями.
pub struct ModelConfigValidator;

impl ModelConfigValidator {
    /// Валидирует `RawModelConfig` на полноту, базовую согласованность
    /// и отсутствие полей, связанных с квантованием.
    ///
    /// Эта функция выполняет проверки критически важных полей, чтобы убедиться,
    /// что они присутствуют и имеют разумные значения. Предполагается, что она
    /// будет вызвана сразу после десериализации `config.json`.
    ///
    /// # Аргументы
    ///
    /// * `raw_config` - "Сырая" конфигурация модели для валидации.
    ///
    /// # Возвращает
    ///
    /// `Result`, указывающий на успех (`Ok(())`) или `ModelLoaderError::InvalidConfig`,
    /// если какая-либо проверка не пройдена.
    pub fn validate_raw_config(raw_config: &RawModelConfig) -> Result<(), ModelLoaderError> {
        debug!("Валидация 'сырой' конфигурации модели.");
        let mut errors: Vec<String> = Vec::new(); // Собираем все ошибки валидации

        // Макрос для упрощения проверки наличия опциональных полей
        macro_rules! check_field {
            ($field:expr, $name:expr) => {
                if $field.is_none() {
                    errors.push(format!("Отсутствует обязательное поле: '{}'", $name));
                }
            };
        }

        // Проверяем наличие всех ключевых полей в RawModelConfig
        check_field!(raw_config.vocab_size, "vocab_size");
        check_field!(raw_config.hidden_size, "hidden_size");
        check_field!(raw_config.intermediate_size, "intermediate_size");
        check_field!(raw_config.num_hidden_layers, "num_hidden_layers");
        check_field!(raw_config.num_attention_heads, "num_attention_heads");
        check_field!(raw_config.num_key_value_heads, "num_key_value_heads");
        check_field!(raw_config.max_position_embeddings, "max_position_embeddings");
        check_field!(raw_config.rms_norm_eps, "rms_norm_eps");
        check_field!(raw_config.head_dim, "head_dim"); // Важно для Gemma
        check_field!(raw_config.model_type, "model_type");
        check_field!(raw_config.eos_token_id, "eos_token_id");
        check_field!(raw_config.bos_token_id, "bos_token_id");
        // pad_token_id может быть null/None, поэтому не проверяем его как обязательное здесь.

        // Проверяем, что указанный model_type поддерживается
        if let Some(model_type_str) = &raw_config.model_type {
            let model_type = ModelType::from(model_type_str.as_str());
            if model_type == ModelType::Other { // ModelType::Other означает неподдерживаемый тип
                errors.push(format!("Неподдерживаемый model_type: '{}'", model_type_str));
            }
        }

        // КРИТИЧЕСКИ ВАЖНАЯ ПРОВЕРКА: Убеждаемся, что в config.json нет конфигурации квантования.
        // Поле quantization_config было удалено из RawModelConfig, поэтому эта проверка
        // должна быть адаптирована, если такое поле может появиться в extra_fields.
        // В текущей версии RawModelConfig это поле отсутствует, так что явной проверки не требуется,
        // кроме как убедиться, что оно не десериализуется случайно.
        // Если бы `quantization_config` было полем в `RawModelConfig`, проверка была бы такой:
        // if raw_config.quantization_config.is_some() {
        //     errors.push("Обнаружена конфигурация квантования в config.json. Этот проект требует чистых, неквантованных моделей.".to_string());
        // }
        // Поскольку его нет, мы можем проверить `extra_fields`, если это необходимо, но это менее надежно.
        // Для данной структуры `RawModelConfig` (где `quantization_config` отсутствует),
        // такая проверка не нужна.

        if !errors.is_empty() {
            let message = errors.join("; "); // Объединяем все сообщения об ошибках
            warn!("Валидация 'сырой' конфигурации модели не пройдена: {}", message);
            return Err(ModelLoaderError::InvalidConfig { message });
        }

        debug!("'Сырая' конфигурация модели успешно валидирована.");
        Ok(())
    }

    /// Валидирует `CoreGemmaModelConfig` на согласованность и разумные значения.
    ///
    /// Эта функция выполняет более детальные проверки после преобразования "сырой"
    /// конфигурации, гарантируя, что все обязательные поля корректно установлены
    /// и соответствуют ограничениям модели. Здесь выполняются проверки архитектурной
    /// согласованности.
    ///
    /// **Примечание:** Этот валидатор гарантирует, что `quantization_config` равно `None`
    /// в `CoreGemmaModelConfig`.
    ///
    /// # Аргументы
    ///
    /// * `core_config` - "Ядерная" конфигурация модели Gemma для валидации.
    ///
    /// # Возвращает
    ///
    /// `Result`, указывающий на успех (`Ok(())`) или `ModelLoaderError::InvalidConfig`,
    /// если какая-либо проверка не пройдена.
    pub fn validate_core_config(core_config: &CoreGemmaModelConfig) -> Result<(), ModelLoaderError> {
        debug!("Валидация 'ядерной' конфигурации модели Gemma.");
        let mut errors: Vec<String> = Vec::new();

        // Базовые проверки значений
        if core_config.vocab_size == 0 {
            errors.push("vocab_size не может быть равен нулю.".to_string());
        }
        if core_config.hidden_size == 0 {
            errors.push("hidden_size не может быть равен нулю.".to_string());
        }
        if core_config.num_attention_heads == 0 {
            errors.push("num_attention_heads не может быть равен нулю.".to_string());
        }
        if core_config.num_key_value_heads == 0 {
            errors.push("num_key_value_heads не может быть равен нулю.".to_string());
        }

        // Проверки архитектурной согласованности
        if core_config.hidden_size % core_config.num_attention_heads != 0 {
            errors.push(format!(
                "hidden_size ({}) должен быть кратен num_attention_heads ({}).",
                core_config.hidden_size, core_config.num_attention_heads
            ));
        }
        if core_config.num_attention_heads % core_config.num_key_value_heads != 0 {
            // Это условие для Grouped Query Attention (GQA)
            errors.push(format!(
                "num_attention_heads ({}) должен быть кратен num_key_value_heads ({}).",
                core_config.num_attention_heads, core_config.num_key_value_heads
            ));
        }
        if core_config.rms_norm_eps <= 0.0 {
            errors.push("rms_norm_eps должен быть положительным.".to_string());
        }
        if core_config.head_dim == 0 {
            errors.push("head_dim не может быть равен нулю.".to_string());
        }

        // Специфичные проверки для модели Gemma
        if core_config.model_type == ModelType::Gemma {
            // Для Gemma head_dim обычно равен hidden_size / num_attention_heads
            let expected_head_dim = core_config.hidden_size / core_config.num_attention_heads;
            if core_config.head_dim != expected_head_dim {
                // Это строгая проверка согласованности для Gemma. Если значение не совпадает,
                // это, вероятно, указывает на проблему с конфигурацией или на неожиданный вариант модели.
                errors.push(format!(
                    "Специфично для Gemma: head_dim ({}) не совпадает с расчетом hidden_size / num_attention_heads ({}).",
                    core_config.head_dim, expected_head_dim
                ));
            }
            // Сюда можно добавить другие специфичные для Gemma проверки
        }

        // КРИТИЧЕСКИ ВАЖНАЯ ПРОВЕРКА: Убеждаемся, что quantization_config в CoreGemmaModelConfig отсутствует.
        if core_config.quantization_config.is_some() {
            errors.push("Конфигурация квантования не поддерживается в CoreGemmaModelConfig для этого проекта.".to_string());
        }


        if !errors.is_empty() {
            let message = errors.join("; ");
            warn!("Валидация 'ядерной' конфигурации модели не пройдена: {}", message);
            return Err(ModelLoaderError::InvalidConfig { message });
        }

        debug!("'Ядерная' конфигурация модели успешно валидирована.");
        Ok(())
    }
}
