use burn::record::{
    // PrecisionSettings, // Не используется напрямую здесь
    QuantizationSettings, // Все еще используется для CoreGemmaModelConfig, но будет всегда None
};
use serde::{
    Deserialize, // Для десериализации из JSON
    // Deserializer, // Не используется напрямую здесь
};
use std::collections::HashMap; // Для поля `extra_fields` в RawModelConfig
use std::fmt; // Для реализации типажа Display для ModelType

/// Представляет "сырую" конфигурацию модели, как она десериализуется напрямую из `config.json`.
///
/// Эта структура разработана гибкой, допуская опциональные поля, которые могут
/// присутствовать или отсутствовать в зависимости от конкретного варианта модели.
///
/// **Примечание:** Эта структура не поддерживает конфигурации квантования,
/// так как проект работает с чистыми, неквантованными моделями SafeTensors.
#[derive(Debug, Clone, Deserialize)]
pub struct RawModelConfig {
    // Основные параметры архитектуры модели
    pub vocab_size: Option<usize>, // Размер словаря
    pub hidden_size: Option<usize>, // Размер скрытого слоя
    pub intermediate_size: Option<usize>, // Размер промежуточного слоя (в MLP)
    pub num_hidden_layers: Option<usize>, // Количество скрытых слоев
    pub num_attention_heads: Option<usize>, // Количество голов внимания
    pub num_key_value_heads: Option<usize>, // Количество голов для ключей/значений (для Grouped Query Attention)
    pub max_position_embeddings: Option<usize>, // Максимальная длина последовательности
    pub rms_norm_eps: Option<f64>, // Эпсилон для RMS нормализации (используем f64 для десериализации из JSON)
    pub rope_theta: Option<f64>,   // Параметр theta для RoPE (Rotary Position Embedding) (f64 из JSON)
    pub use_cache: Option<bool>, // Использовать ли кэш предыдущих состояний (для ускорения генерации)
    pub torch_dtype: Option<String>, // Тип данных, используемый в PyTorch (например, "float32", "bfloat16")
    pub model_type: Option<String>, // Тип модели (например, "gemma", "llama")
    pub eos_token_id: Option<usize>, // ID токена конца последовательности
    pub bos_token_id: Option<usize>, // ID токена начала последовательности
    pub pad_token_id: Option<usize>, // ID токена для паддинга (может быть null)
    pub head_dim: Option<usize>, // Размерность каждой головы внимания (характерно для Gemma)

    // Поле quantization_config было удалено, так как оно не используется.
    // pub quantization_config: Option<RawQuantizedModelConfig>,

    // Поле для сбора всех остальных полей из JSON, которые не были явно определены выше.
    // Это позволяет config.json содержать дополнительные метаданные без ошибок при парсинге.
    #[serde(flatten)]
    pub extra_fields: HashMap<String, serde_json::Value>,
}

// Структура RawQuantizedModelConfig и ее реализация были удалены.

/// Представляет "ядерную" конфигурацию модели Gemma, адаптированную для конструктора моделей Burn.
///
/// Все поля здесь ожидаются присутствующими и корректно типизированными для инстанцирования модели.
///
/// **Примечание:** Поле `quantization_config` всегда будет `None`, так как этот загрузчик
/// поддерживает только неквантованные модели. Оно сохранено как `Option` для совместимости
/// с типажом `Config` из Burn для моделей, который часто включает это поле.
#[derive(Debug, Clone, PartialEq)] // PartialEq добавлен для удобства тестирования
pub struct CoreGemmaModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32, // Используем f32 для "ядерной" модели
    pub rope_theta: f32,   // Используем f32 для "ядерной" модели
    pub use_cache: bool,
    pub torch_dtype: Option<String>, // Может быть None, если не указано
    pub quantization_config: Option<QuantizationSettings>, // Будет всегда None в этом проекте
    pub head_dim: usize, // Обязательно для Gemma
    pub eos_token_id: usize, // Обязательно для токенизации
    pub bos_token_id: usize, // Обязательно для токенизации
    pub pad_token_id: Option<usize>, // Может быть None
    pub model_type: ModelType, // Обязательно, разобранное перечисление
}

/// Представляет запись (веса) Burn для модели Gemma, используемую для загрузки весов.
///
/// Это псевдоним типа для обобщенного типа `Record` из Burn, специально
/// для модели Gemma с заданным бэкендом `B`.
/// Изменено для использования `core_burn::GemmaModel` согласно зависимости в `Cargo.toml`.
/// `GemmaModelRecord<B>` фактически является структурой, которую Burn генерирует
/// на основе полей `Param<Tensor<...>>` в `core_burn::GemmaModel`.
pub type GemmaModelRecord<B> = burn::record::Record<
    <core_burn::GemmaModel<B> as burn::module::Module>::Record,
>;


/// Перечисляет поддерживаемые типы моделей.
#[derive(Debug, Clone, PartialEq)] // PartialEq для сравнения в тестах и логике
pub enum ModelType {
    Gemma, // Модель Gemma
    Llama, // Модель Llama
    Phi,   // Модель Phi
    Other, // Для неподдерживаемых или неизвестных типов моделей
}

// Реализация конвертации из строкового среза в ModelType.
// Позволяет легко парсить поле "model_type" из config.json.
impl From<&str> for ModelType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() { // Приводим к нижнему регистру для нечувствительности к регистру
            "gemma" => ModelType::Gemma,
            "llama" => ModelType::Llama,
            "phi"   => ModelType::Phi,
            _       => ModelType::Other, // Все остальное считаем неподдерживаемым
        }
    }
}

// Реализация типажа Display для ModelType для удобного вывода.
impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelType::Gemma => write!(f, "gemma"),
            ModelType::Llama => write!(f, "llama"),
            ModelType::Phi   => write!(f, "phi"),
            ModelType::Other => write!(f, "other"),
        }
    }
}

/// Информация о загруженной модели, объединяющая ее тип и конфигурацию.
/// (В текущей реализации `loader.rs` эта структура не возвращается напрямую,
/// но может быть полезна для вышестоящих крейтов).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: ModelType, // Тип модели
    pub config: CoreGemmaModelConfig, // "Ядерная" конфигурация
    // Сюда можно добавить другую релевантную информацию, такую как путь к токенизатору и т.д.
}

/// Представляет текст, который должен быть добавлен к промпту.
///
/// Это может быть либо прямая строка, либо путь к файлу, содержащему текст.
#[derive(Debug, Clone, Deserialize, PartialEq)] // PartialEq для тестов
#[serde(untagged)] // Позволяет десериализацию либо из строки, либо из объекта JSON
pub enum AppendedText {
    /// Прямое строковое содержимое для добавления.
    Text(String),
    /// Путь к файлу, содержащему текст для добавления.
    FilePath { path: String },
}
