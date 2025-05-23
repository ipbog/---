#![warn(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used
)]
#![deny(unsafe_code, unused_mut, unused_imports, unused_attributes)]

#[cfg(feature = "inference_types_serde")]
use serde::{Deserialize, Serialize};
use super::sampling_params::SamplingParams;

#[cfg(feature = "with_tokio_sender")]
use tokio::sync::mpsc::Sender;

#[cfg(all(
    feature = "inference_types_serde",
    feature = "either"
))]
use either::Either;
#[cfg(all(
    feature = "inference_types_serde",
    feature = "indexmap"
))]
use indexmap::IndexMap;
#[cfg(all(
    feature = "inference_types_serde",
    feature = "serde_json"
))]
use serde_json::Value;

/// Содержимое сообщения, может быть простым текстом или структурированным списком для мультимодальности.
#[cfg(all(
    feature = "inference_types_serde",
    feature = "either",
    feature = "indexmap",
    feature = "serde_json"
))]
pub type MessageContent = Either<String, Vec<IndexMap<String, Value>>>;

/// Содержимое сообщения (версия без сложных типов, если фичи не включены).
#[cfg(not(all(
    feature = "inference_types_serde",
    feature = "either",
    feature = "indexmap",
    feature = "serde_json"
)))]
pub type MessageContent = String;

/// Представляет одно сообщение в чате.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub struct ChatMessage {
    /// Роль отправителя сообщения (например, "user", "assistant", "system").
    pub role: String,
    /// Содержимое сообщения.
    pub content: MessageContent,
}

/// Входные данные для задачи инференса.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub enum TaskInput {
    /// Завершение текста с опциональным возвратом промпта.
    Completion {
        /// Текст для завершения.
        text: String,
        /// Возвращать ли промпт вместе с результатом.
        #[cfg_attr(feature = "inference_types_serde", serde(default))]
        echo_prompt: bool,
    },
    /// Взаимодействие в формате чата со списком сообщений.
    Chat {
        /// Список сообщений чата.
        messages: Vec<ChatMessage>,
    },
    /// Прямая передача ID токенов для завершения (для продвинутого использования).
    CompletionTokens(Vec<u32>),
}

/// Метрики использования токенов для данного инференса.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
pub struct UsageMetrics {
    /// Количество токенов во входном промпте/сообщениях.
    pub prompt_tokens: usize,
    /// Количество сгенерированных токенов в ответе.
    pub completion_tokens: usize,
    /// Общее количество токенов (промпт + завершение).
    pub total_tokens: usize,
}

/// Представляет чанк потокового ответа инференса.
/// Отправляется многократно для одной задачи, если `stream_response` равно `true`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "inference_types_serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "inference_types_serde", serde(untagged))] // Позволяет гибкую десериализацию
pub enum InferenceResponse {
    /// Событие с данными, содержащее сгенерированный текст и, возможно, метрики использования.
    Data {
        /// ID запроса.
        request_id: String,
        /// Чанк текста.
        text_chunk: String,
        // logprobs: Option<Vec<(String, f32)>>, // Пока закомментировано
        /// Является ли это последним чанком.
        is_final_chunk: bool,
        /// Метрики использования (могут обновляться с каждым чанком или быть только в конце).
        usage: Option<UsageMetrics>,
    },
    /// Финальный ответ, содержащий полный текст и метрики использования.
    Done {
        /// ID запроса.
        request_id: String,
        /// Полный сгенерированный текст.
        full_text: String,
        // logprobs: Option<Vec<(String, f32)>>, // Пока закомментировано
        /// Итоговые метрики использования.
        usage: UsageMetrics,
        /// Причина остановки генерации.
        stop_reason: String,
    },
    /// Специальное событие для ошибки.
    Error {
        /// ID запроса.
        request_id: String,
        /// Сообщение об ошибке.
        error: String,
    },
}

/// Представляет запрос на выполнение задачи инференса.
#[derive(Debug)]
pub struct InferenceTask {
    /// Уникальный идентификатор задачи.
    pub id: String,
    /// Входные данные для инференса (промпт, сообщения или ID токенов).
    pub input: TaskInput,
    /// Параметры для управления процессом сэмплирования.
    pub sampling_params: SamplingParams,
    /// Максимальное количество новых токенов для генерации.
    pub max_new_tokens: usize,
    /// Последовательности токенов, которые остановят генерацию (переопределяют `SamplingParams`, если предоставлены).
    pub stop_sequences: Option<Vec<String>>,
    /// Возвращать ли ответ потоково, токен за токеном.
    pub stream_response: bool,
    /// Возвращать ли лог-вероятности для сгенерированных токенов.
    pub return_logprobs: bool,
    /// Канал для отправки потоковых ответов инференса обратно вызывающей стороне.
    #[cfg(feature = "with_tokio_sender")]
    pub response_sender: Sender<InferenceResponse>,
}

#[cfg(feature = "with_tokio_sender")]
impl InferenceTask {
    /// Создает новый экземпляр `InferenceTask`.
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
