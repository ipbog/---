#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

//! Модуль определяет структуры данных, связанные с задачами инференса.

#[cfg(feature = "inference_types_serde")]
use serde::{Deserialize, Serialize};
use super::sampling_params::SamplingParams;

#[cfg(feature = "with_tokio_sender")]
use tokio::sync::mpsc::Sender;

#[cfg(all(feature = "inference_types_serde", feature = "either"))]
use either::Either;
#[cfg(all(feature = "inference_types_serde", feature = "indexmap"))]
use indexmap::IndexMap;
#[cfg(all(feature = "inference_types_serde", feature = "serde_json"))]
use serde_json::Value;

/// Содержимое сообщения: простой текст или структурированный список.
///
/// Активируется фичами `inference_types_serde`, `either`, `indexmap`, `serde_json`.
#[cfg(all(
    feature = "inference_types_serde",
    feature = "either",
    feature = "indexmap",
    feature = "serde_json"
))]
pub type MessageContent = Either<String, Vec<IndexMap<String, Value>>>;

/// Содержимое сообщения (упрощенная версия, если не все фичи для сложного типа включены).
#[cfg(not(all(
    feature = "inference_types_serde",
    feature = "either",
    feature = "indexmap",
    feature = "serde_json"
)))]
pub type MessageContent = String;


/// Одно сообщение в диалоге.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub struct ChatMessage {
    /// Роль отправителя ("user", "assistant", "system").
    pub role: String,
    /// Содержимое сообщения.
    pub content: MessageContent,
}

/// Входные данные для задачи инференса.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub enum TaskInput {
    /// Завершение текста.
    Completion {
        /// Входной текст (промпт).
        text: String,
        /// Включить ли промпт в ответ.
        #[cfg_attr(feature = "inference_types_serde", serde(default))]
        echo_prompt: bool,
    },
    /// Чат.
    Chat {
        /// Список сообщений диалога.
        messages: Vec<ChatMessage>,
    },
    /// Завершение по токенам (расширенный вариант).
    CompletionTokens(Vec<u32>),
}

/// Метрики использования токенов.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub struct UsageMetrics {
    /// Токены промпта.
    pub prompt_tokens: usize,
    /// Сгенерированные токены.
    pub completion_tokens: usize,
    /// Всего токенов.
    pub total_tokens: usize,
}

/// Часть потокового ответа или полный ответ инференса.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "inference_types_serde", serde(untagged))]
pub enum InferenceResponse {
    /// Событие с данными (чанк).
    Data {
        /// ID запроса.
        request_id: String,
        /// Фрагмент текста.
        text_chunk: String,
        /// Является ли это последним чанком данных перед `Done`.
        is_final_chunk: bool,
        /// Промежуточные метрики.
        usage: Option<UsageMetrics>,
    },
    /// Финальный ответ.
    Done {
        /// ID запроса.
        request_id: String,
        /// Полный текст.
        full_text: String,
        /// Итоговые метрики.
        usage: UsageMetrics,
        /// Причина остановки.
        stop_reason: String,
    },
    /// Ошибка инференса.
    Error {
        /// ID запроса.
        request_id: String,
        /// Описание ошибки.
        error: String,
    },
}

/// Запрос на выполнение задачи инференса.
#[derive(Debug)] // PartialEq и Clone не реализуются автоматически из-за Sender, если он не Clone.
pub struct InferenceTask {
    /// Уникальный ID задачи.
    pub id: String,
    /// Входные данные.
    pub input: TaskInput,
    /// Параметры семплирования.
    pub sampling_params: SamplingParams,
    /// Максимальное количество новых токенов.
    pub max_new_tokens: usize,
    /// Последовательности для остановки генерации.
    pub stop_sequences: Option<Vec<String>>,
    /// Возвращать ли ответ потоково.
    pub stream_response: bool,
    /// Возвращать ли логарифмы вероятностей.
    pub return_logprobs: bool,
    /// Канал для отправки потоковых ответов.
    /// Требует фичу `with_tokio_sender`.
    #[cfg(feature = "with_tokio_sender")]
    pub response_sender: Sender<InferenceResponse>,
}

#[cfg(feature = "with_tokio_sender")]
impl InferenceTask {
    /// Создает новый `InferenceTask`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        input: TaskInput,
        sampling_params: SamplingParams,
        max_new_tokens: usize,
        stop_sequences: Option<Vec<String>>,
        stream_response: bool,
        return_logprobs: bool,
        response_sender: Sender<InferenceResponse>,
    ) -> Self {
        Self {
            id,
            input,
            sampling_params,
            max_new_tokens,
            stop_sequences,
            stream_response,
            return_logprobs,
            response_sender,
        }
    }
}
