// api_server_crate/src/main.rs

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use inference_engine::{
    inference_runner::{InferenceRunner, InferenceRunnerBuilder},
    config::EngineConfig,
    utils_crate::{InferenceTask, SamplingParams},
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tracing::{info, error, debug};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

// Выбираем бэкенд по умолчанию.
#[cfg(feature = "ndarray_backend")]
type Backend = burn::backend::ndarray::NdArray;
#[cfg(feature = "tch_backend")]
type Backend = burn::backend::tch::Tch;
#[cfg(feature = "wgpu_backend")]
type Backend = burn::backend::wgpu::Wgpu;

// Структура для запроса на инференс
#[derive(Debug, Deserialize)]
struct InferenceRequest {
    prompt: String,
    #[serde(default)]
    temperature: f32,
    #[serde(default)]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    seed: Option<u64>,
}

// Структура для ответа инференса
#[derive(Debug, Serialize)]
struct InferenceResponse {
    generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// Структура для Application State, которая будет содержать InferenceRunner
#[derive(Clone)]
struct AppState<B: Backend> {
    runner: Arc<InferenceRunner<B>>,
}

// Async-функция для обработки запросов на инференс
async fn generate(
    State(state): State<AppState<Backend>>,
    Json(payload): Json<InferenceRequest>,
) -> Json<InferenceResponse> {
    info!("Received inference request: {:?}", payload);

    let sampling_params = SamplingParams {
        temperature: payload.temperature,
        top_p: payload.top_p,
        top_k: payload.top_k,
        seed: payload.seed,
    };

    let task = InferenceTask {
        prompt: payload.prompt,
        sampling_params,
    };

    match state.runner.generate_text(task).await {
        Ok(text) => {
            info!("Text generation successful.");
            Json(InferenceResponse {
                generated_text: text,
                error: None,
            })
        }
        Err(e) => {
            error!("Text generation failed: {}", e);
            Json(InferenceResponse {
                generated_text: String::new(),
                error: Some(e.to_string()),
            })
        }
    }
}

// Добавим простую проверку работоспособности сервера
async fn health_check() -> &'static str {
    "OK"
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Настройка логирования
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(tracing::Level::INFO) // По умолчанию INFO, можно изменить через RUST_LOG
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    // --- Параметры инициализации модели (можно сделать через env vars или конфиг) ---
    // Для примера, используем жестко заданные пути
    let model_path = PathBuf::from("./models/gemma-2b-it"); // Пример пути
    let tokenizer_path = PathBuf::from("./models/gemma-2b-it/tokenizer.json"); // Пример пути
    let model_name = "gemma-2b-it".to_string();
    let model_type = "gemma".to_string();
    let cache_dir = ".model_cache".to_string();
    // --------------------------------------------------------------------------

    let device = Backend::Device::default(); // NdArrayDevice::Cpu;

    let engine_config = EngineConfig {
        max_kv_cache_length: 2048,
        max_generation_length: 256,
        load_quantized: false,
        model_cache_dir: cache_dir,
    };

    info!("Initializing InferenceRunner for API server...");
    debug!("Using device: {:?}", device);

    let runner = InferenceRunnerBuilder::new()
        .with_config(engine_config)
        .with_model_path(model_path)
        .with_tokenizer_path(tokenizer_path)
        .with_model_name(model_name)
        .with_model_type(model_type)
        .build::<Backend>(device)
        .await;

    let runner = match runner {
        Ok(r) => {
            info!("InferenceRunner initialized successfully for API server.");
            Arc::new(r) // Оборачиваем в Arc для совместного использования между хэндлерами
        }
        Err(e) => {
            error!("Failed to initialize InferenceRunner for API server: {}", e);
            // Если не удалось инициализировать движок, сервер не должен запускаться.
            return Err(e.into());
        }
    };

    let app_state = AppState { runner };

    // Построение маршрутов Axum
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/generate", post(generate))
        .with_state(app_state);

    // Запуск сервера
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    info!("API server listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```
