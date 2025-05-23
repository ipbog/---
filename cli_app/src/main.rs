// cli_app/src/main.rs

use clap::Parser;
use inference_engine::{
    inference_runner::{InferenceRunner, InferenceRunnerBuilder},
    config::EngineConfig,
    utils_crate::{InferenceTask, SamplingParams},
};
use burn::backend::ndarray::NdArrayDevice; // Использование NdArray в качестве бэкенда по умолчанию
use std::path::PathBuf;
use tracing::{info, error, debug};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

// Выбираем бэкенд по умолчанию. Можно добавить условную компиляцию для других бэкендов.
#[cfg(feature = "ndarray_backend")]
type Backend = burn::backend::ndarray::NdArray;
#[cfg(feature = "tch_backend")]
type Backend = burn::backend::tch::Tch;
#[cfg(feature = "wgpu_backend")]
type Backend = burn::backend::wgpu::Wgpu;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
/// Простой CLI-интерфейс для запуска инференса моделей.
struct Args {
    /// Путь к директории с файлами модели (config.json, model.safetensors).
    #[clap(long, value_parser)]
    model_path: PathBuf,

    /// Путь к файлу токенизатора (tokenizer.json).
    #[clap(long, value_parser)]
    tokenizer_path: PathBuf,

    /// Имя модели (например, "gemma-7b-it").
    #[clap(long, default_value = "gemma-default")]
    model_name: String,

    /// Тип архитектуры модели (например, "gemma").
    #[clap(long, default_value = "gemma")]
    model_type: String,

    /// Промпт для генерации текста.
    #[clap(long, value_parser)]
    prompt: String,

    /// Температура сэмплирования. Установите 0 для детерминированного выхода (argmax).
    #[clap(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-P (ядерное) сэмплирование. От 0.0 до 1.0.
    #[clap(long, default_value_t = 0.9)]
    top_p: f32,

    /// Top-K сэмплирование. 0 для отключения.
    #[clap(long, default_value_t = 0)]
    top_k: usize,

    /// Случайный сид для сэмплирования для воспроизводимости.
    #[clap(long)]
    seed: Option<u64>,

    /// Максимальная длина KV-кэша.
    #[clap(long, default_value_t = 2048)]
    max_kv_cache_length: usize,

    /// Максимальная длина генерируемого текста.
    #[clap(long, default_value_t = 256)]
    max_generation_length: usize,

    /// Использовать квантованные модели, если доступны.
    #[clap(long, action)]
    quantized: bool,

    /// Директория для кэширования моделей.
    #[clap(long, default_value = ".model_cache")]
    cache_dir: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Настройка логирования
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(tracing::Level::INFO) // По умолчанию INFO, можно изменить через RUST_LOG
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let args = Args::parse();
    info!("Parsed arguments: {:?}", args);

    // Определяем устройство. Для NdArray это всегда CPU.
    // Для Tch/Wgpu здесь можно выбрать GPU, если доступно.
    let device = Backend::Device::default(); // NdArrayDevice::Cpu;

    let engine_config = EngineConfig {
        max_kv_cache_length: args.max_kv_cache_length,
        max_generation_length: args.max_generation_length,
        load_quantized: args.quantized,
        model_cache_dir: args.cache_dir,
    };

    info!("Initializing InferenceRunner with config: {:?}", engine_config);
    debug!("Using device: {:?}", device);

    let runner = InferenceRunnerBuilder::new()
        .with_config(engine_config)
        .with_model_path(args.model_path)
        .with_tokenizer_path(args.tokenizer_path)
        .with_model_name(args.model_name)
        .with_model_type(args.model_type)
        .build::<Backend>(device)
        .await;

    let runner = match runner {
        Ok(r) => {
            info!("InferenceRunner initialized successfully.");
            r
        }
        Err(e) => {
            error!("Failed to initialize InferenceRunner: {}", e);
            return Err(e.into());
        }
    };

    let sampling_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
    };

    let task = InferenceTask {
        prompt: args.prompt,
        sampling_params,
    };

    info!("Starting text generation for prompt: '{}'", task.prompt);
    debug!("Sampling parameters: {:?}", task.sampling_params);

    let generated_text = runner.generate_text(task).await;

    match generated_text {
        Ok(text) => {
            println!("
--- Generated Text ---");
            println!("{}", text);
            println!("----------------------");
            info!("Text generation completed successfully.");
        }
        Err(e) => {
            error!("Text generation failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
```
