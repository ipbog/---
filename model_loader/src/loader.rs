use std::collections::HashMap; // Используется для `extra_fields` в `RawModelConfig`, здесь не напрямую
use std::path::{Path, PathBuf}; // Для работы с путями к файлам

use burn::{
    // module::Param, // Не используется напрямую в этом файле после изменений
    record::{
        // record, // Не используется напрямую
        FileRecorder, // Для загрузки весов из файлов
        FullPrecisionSettings, // Всегда используем полную точность, квантование не поддерживается
        Recorder, // Общий типаж для рекордеров
    },
    tensor::{
        // tensor, // Не используется напрямую
        backend::Backend, // Типаж для бэкенда Burn (например, NdArray, Wgpu)
        // container::TensorContainer, // Не используется напрямую
        // DType, // Не используется напрямую
        // Data, // Не используется напрямую
        // Tensor, // Не используется напрямую
    },
};
// use serde::{ // Serde используется в `types.rs` для десериализации
//     de::Error as SerdeError,
//     Deserialize,
//     Deserializer,
// };
use tracing::{ // Для логирования
    debug, // Уровень отладки
    info, // Информационный уровень
    warn, // Уровень предупреждений
};

// Условная компиляция для интеграции ошибок с utils_crate
#[cfg(feature = "with_utils_crate_errors")]
use utils_crate::errors::ModelHubError; // Предполагается, что такой тип ошибки существует в utils_crate

use crate::{
    error::ModelLoaderError, // Кастомные ошибки загрузчика
    types::{ // Типы данных, определенные в этом крейте
        // AppendedText, // Не используется напрямую в этом файле
        CoreGemmaModelConfig, // Основная ("ядерная") конфигурация модели Gemma
        GemmaModelRecord, // Тип записи (весов) для модели Gemma
        // ModelInfo, // Не используется напрямую в этом файле после изменений
        ModelType, // Перечисление типов моделей (Gemma, Llama и т.д.)
        RawModelConfig, // "Сырая" конфигурация, как она есть в config.json
        // RawQuantizedModelConfig - удален, так как квантование не поддерживается
    },
    validation::ModelConfigValidator, // Валидатор конфигураций
};

/// Структура для загрузки моделей Машинного Обучения, специально разработанная для
/// моделей Gemma и совместимая с экосистемой Burn. Загружает модели из локальных
/// sharded (разделенных на части) файлов SafeTensors.
///
/// Этот загрузчик строго обрабатывает парсинг конфигурации и загрузку весов
/// из sharded SafeTensors. Он работает в оффлайн-режиме и не поддерживает
/// квантованные модели или сетевые операции.
#[derive(Debug, Clone)]
pub struct ModelLoader;

impl ModelLoader {
    /// Загружает модель из указанной директории.
    ///
    /// Эта функция обрабатывает полный жизненный цикл загрузки модели:
    /// 1. Загружает и разбирает "сырую" конфигурацию из `config.json`.
    /// 2. Валидирует "сырую" конфигурацию с помощью `ModelConfigValidator`.
    /// 3. Преобразует "сырую" конфигурацию в `CoreGemmaModelConfig`.
    /// 4. Валидирует "ядерную" конфигурацию.
    /// 5. Находит и загружает sharded файлы весов safetensors, используя `model.safetensors.index.json`.
    /// 6. Создает `GemmaModelRecord` из загруженных тензоров.
    ///
    /// # Аргументы
    ///
    /// * `model_dir` - Директория, содержащая `config.json` модели и sharded файлы весов.
    /// * `device` - Устройство Burn, на которое будет загружена модель.
    ///
    /// # Возвращает
    ///
    /// `Result`, содержащий загруженную `GemmaModelRecord` (веса) и ее `CoreGemmaModelConfig`
    /// в случае успеха, или `ModelLoaderError` в случае сбоя.
    pub async fn load_model<B: Backend>(
        model_dir: &Path,
        device: &B::Device,
    ) -> Result<(GemmaModelRecord<B>, CoreGemmaModelConfig), ModelLoaderError> {
        info!("Начало загрузки модели из: {:?}", model_dir);

        // Формируем путь к файлу config.json
        let config_path = model_dir.join("config.json");
        // Асинхронно загружаем и парсим "сырую" конфигурацию
        let raw_config: RawModelConfig = Self::load_json_config(&config_path).await?;

        // Немедленно валидируем "сырую" конфигурацию после загрузки
        ModelConfigValidator::validate_raw_config(&raw_config)?;

        // Извлекаем тип модели из "сырой" конфигурации
        let model_type_str = raw_config
            .model_type // Поле model_type в RawModelConfig теперь Option<String>
            .clone() // Клонируем строку, если она есть
            .ok_or_else(|| ModelLoaderError::ConfigParsing { // Если None, возвращаем ошибку
                path: config_path.display().to_string(),
                source: "Отсутствует поле 'model_type'".into(), // Используем .into() для Box<dyn Error>
            })?;
        let model_type = ModelType::from(model_type_str.as_str()); // Преобразуем строку в enum ModelType

        info!("Обнаружен тип модели: {:?}", model_type);

        // Конвертируем "сырую" конфигурацию в "ядерную"
        let core_config = Self::convert_raw_config_to_core_config(&raw_config)?;

        // Валидируем "ядерную" конфигурацию после конвертации
        ModelConfigValidator::validate_core_config(&core_config)?;

        // Всегда загружаем sharded модель, так как это требование проекта
        info!("Загрузка весов sharded модели из: {:?}", model_dir);
        let model_record = Self::load_sharded_model_record::<B>(model_dir, device).await?;

        info!("Загрузка модели завершена.");
        Ok((model_record, core_config))
    }

    /// Загружает конфигурационный файл JSON из указанного пути.
    ///
    /// Эта функция выполняет асинхронное чтение файла и десериализацию.
    ///
    /// # Аргументы
    ///
    /// * `path` - Путь к файлу конфигурации JSON.
    ///
    /// # Возвращает
    ///
    /// `Result`, содержащий десериализованную `RawModelConfig` в случае успеха,
    /// или `ModelLoaderError` в случае сбоя (ошибки ввода/вывода или парсинга).
    async fn load_json_config(path: &Path) -> Result<RawModelConfig, ModelLoaderError> {
        debug!("Загрузка JSON конфигурации из: {:?}", path);
        // Асинхронно открываем файл
        let file = tokio::fs::File::open(path)
            .await
            .map_err(|e| ModelLoaderError::Io { // В случае ошибки открытия, возвращаем нашу кастомную ошибку
                path: path.display().to_string(),
                source: e,
            })?;

        // Создаем буферизованный ридер для асинхронного файла
        let reader = tokio::io::BufReader::new(file);
        // Десериализуем JSON из ридера. `into_std()` конвертирует асинхронный ридер в стандартный,
        // так как `serde_json` работает со стандартными ридерами.
        let config: RawModelConfig = serde_json::from_reader(reader.into_std()).map_err(
            |e| ModelLoaderError::ConfigParsing {
                path: path.display().to_string(),
                source: Box::new(e), // Оборачиваем ошибку serde_json в Box для ModelLoaderError
            },
        )?;

        debug!("JSON конфигурация успешно загружена.");
        Ok(config)
    }

    /// Преобразует `RawModelConfig` в `CoreGemmaModelConfig`.
    ///
    /// Эта функция обрабатывает сопоставление полей и возможные значения по умолчанию
    /// для параметров модели. Теперь она гарантирует, что все обязательные поля для
    /// `CoreGemmaModelConfig` присутствуют или выведены.
    ///
    /// # Аргументы
    ///
    /// * `raw_config` - "Сырая" конфигурация модели, полученная из `config.json`.
    ///
    /// # Возвращает
    ///
    /// `Result`, содержащий `CoreGemmaModelConfig` в случае успеха,
    /// или `ModelLoaderError`, если обязательное поле отсутствует или невалидно.
    fn convert_raw_config_to_core_config(
        raw_config: &RawModelConfig,
    ) -> Result<CoreGemmaModelConfig, ModelLoaderError> {
        debug!("Конвертация 'сырой' конфигурации в 'ядерную'.");

        // Строка пути для контекста ошибок, если конфигурация создается "в памяти"
        let path_str = "конфигурация из памяти (производная)".to_string();

        // Явно получаем обязательные поля, используя `ok_or_else` для лучших сообщений об ошибках.
        // Если поле None, будет возвращена ошибка `ConfigParsing`.
        let vocab_size = raw_config.vocab_size.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'vocab_size'".into(),
            }
        })?;
        let hidden_size = raw_config.hidden_size.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'hidden_size'".into(),
            }
        })?;
        let intermediate_size = raw_config.intermediate_size.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'intermediate_size'".into(),
            }
        })?;
        let num_hidden_layers = raw_config.num_hidden_layers.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'num_hidden_layers'".into(),
            }
        })?;
        let num_attention_heads = raw_config.num_attention_heads.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'num_attention_heads'".into(),
            }
        })?;
        let num_key_value_heads = raw_config.num_key_value_heads.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'num_key_value_heads'".into(),
            }
        })?;
        let max_position_embeddings = raw_config.max_position_embeddings.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'max_position_embeddings'".into(),
            }
        })?;
        // `rms_norm_eps` загружается как f64, затем конвертируется в f32 для `CoreGemmaModelConfig`.
        let rms_norm_eps = raw_config.rms_norm_eps.map(|v| v as f32).ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'rms_norm_eps'".into(),
            }
        })?;
        let head_dim = raw_config.head_dim.ok_or_else(|| {
            ModelLoaderError::ConfigParsing {
                path: path_str.clone(),
                source: "Отсутствует 'head_dim'".into(),
            }
        })?;

        // Важно: Убеждаемся, что эти обязательные поля правильно обработаны,
        // с значениями по умолчанию, если это уместно и соответствует спецификации модели.
        let eos_token_id = raw_config.eos_token_id.unwrap_or(1); // Значение по умолчанию для Gemma
        let bos_token_id = raw_config.bos_token_id.unwrap_or(2); // Значение по умолчанию для Gemma
        let pad_token_id = raw_config.pad_token_id; // Может быть None для Gemma

        // Тип модели должен быть указан.
        let model_type = raw_config
            .model_type
            .as_ref() // Получаем ссылку на строку, если она есть
            .map(|s| ModelType::from(s.as_str())) // Конвертируем строку в enum ModelType
            .ok_or_else(|| ModelLoaderError::ConfigParsing { // Если model_type None, возвращаем ошибку
                path: path_str.clone(),
                source: "Отсутствует или невалидный 'model_type'".into(),
            })?;

        // Опциональные поля со значениями по умолчанию, если они не указаны в config.json
        let rope_theta = raw_config.rope_theta.unwrap_or(10000.0) as f32; // Конвертируем в f32
        let use_cache = raw_config.use_cache.unwrap_or(true);
        let torch_dtype = raw_config.torch_dtype.clone(); // Клонируем, может быть None

        // Секция quantization_config была удалена, так как квантование не поддерживается.
        // В CoreGemmaModelConfig поле quantization_config будет всегда None.

        let result = CoreGemmaModelConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            use_cache,
            torch_dtype,
            quantization_config: None, // Явно указываем None, так как квантование не поддерживается
            head_dim,
            eos_token_id,
            bos_token_id,
            pad_token_id,
            model_type,
        };

        debug!("'Сырая' конфигурация успешно конвертирована в 'ядерную': {:?}", result);
        Ok(result)
    }

    /// Загружает `GemmaModelRecord` для неквантованной (sharded) модели из нескольких файлов safetensors.
    ///
    /// Эта функция использует `model.safetensors.index.json` для определения, какие тензоры
    /// находятся в каких файлах `model-xxxx-of-yyyy.safetensors`, и загружает их
    /// в единую `GemmaModelRecord`.
    ///
    /// # Аргументы
    ///
    /// * `model_dir` - Директория, содержащая sharded файлы safetensors и `model.safetensors.index.json`.
    /// * `device` - Устройство Burn, на которое будут загружены тензоры.
    ///
    /// # Возвращает
    ///
    /// `Result`, содержащий `GemmaModelRecord` в случае успеха,
    /// или `ModelLoaderError` в случае сбоя.
    async fn load_sharded_model_record<B: Backend>(
        model_dir: &Path,
        device: &B::Device,
    ) -> Result<GemmaModelRecord<B>, ModelLoaderError> {
        info!("Загрузка записи sharded модели из: {:?}", model_dir);

        // Используем FileRecorder из Burn с FullPrecisionSettings, так как квантование не поддерживается.
        let recorder = FileRecorder::<FullPrecisionSettings>::new();
        // Метод load_sharded автоматически найдет `model.safetensors.index.json` в model_dir
        // и загрузит все части модели согласно этому индексу.
        let record = recorder
            .load_sharded(model_dir.to_path_buf(), device) // model_dir должен быть PathBuf
            .map_err(|e| ModelLoaderError::RecordLoading { // В случае ошибки конвертируем ее в нашу
                path: model_dir.display().to_string(),
                source: Box::new(e), // Оборачиваем ошибку Burn в Box
            })?;

        info!("Запись sharded модели успешно загружена.");
        Ok(record)
    }
}
