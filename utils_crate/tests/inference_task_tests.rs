#![cfg(feature = "inference_types_serde")] // Тесты активны только если фича включена

use utils_crate::inference_task::{ChatMessage, TaskInput, UsageMetrics, InferenceResponse, MessageContent};
use utils_crate::sampling_params::SamplingParams; // Для полноты, если понадобится

#[cfg(all(feature = "either", feature = "indexmap", feature = "serde_json"))]
use either::Either;
#[cfg(all(feature = "indexmap", feature = "serde_json"))]
use indexmap::indexmap;
#[cfg(feature = "serde_json")]
use serde_json::json;


#[test]
fn test_chat_message_serialization_deserialization() {
    let original_message = ChatMessage {
        role: "user".to_string(),
        #[cfg(all(feature = "either", feature = "indexmap", feature = "serde_json"))]
        content: MessageContent::Left("Hello, world!".to_string()),
        #[cfg(not(all(feature = "either", feature = "indexmap", feature = "serde_json")))]
        content: "Hello, world!".to_string(),
    };

    let serialized = serde_json::to_string(&original_message).unwrap();
    let deserialized: ChatMessage = serde_json::from_str(&serialized).unwrap();

    assert_eq!(original_message, deserialized);
}

#[cfg(all(feature = "either", feature = "indexmap", feature = "serde_json"))]
#[test]
fn test_chat_message_multimodal_content_serialization() {
    let multimodal_message = ChatMessage {
        role: "user".to_string(),
        content: MessageContent::Right(vec![
            indexmap! { "type".to_string() => json!("text"), "text".to_string() => json!("Hello") },
            indexmap! { "type".to_string() => json!("image_url"), "image_url".to_string() => json!({"url": "http://example.com/image.png"}) },
        ]),
    };

    let serialized = serde_json::to_string(&multimodal_message).unwrap();
    let deserialized: ChatMessage = serde_json::from_str(&serialized).unwrap();
    assert_eq!(multimodal_message, deserialized);

    // Примерный вид JSON
    // {"role":"user","content":[{"type":"text","text":"Hello"},{"type":"image_url","image_url":{"url":"http://example.com/image.png"}}]}
    assert!(serialized.contains("image_url"));
}


#[test]
fn test_task_input_completion_serialization() {
    let task_input = TaskInput::Completion {
        text: "Translate to French: ".to_string(),
        echo_prompt: true,
    };
    let serialized = serde_json::to_string(&task_input).unwrap();
    let deserialized: TaskInput = serde_json::from_str(&serialized).unwrap();
    assert_eq!(task_input, deserialized);
    // Примерный вид JSON: {"Completion":{"text":"Translate to French: ","echo_prompt":true}}
    assert!(serialized.contains("Completion"));
}

#[test]
fn test_task_input_chat_serialization() {
    let task_input = TaskInput::Chat {
        messages: vec![ChatMessage {
            role: "system".to_string(),
            #[cfg(all(feature = "either", feature = "indexmap", feature = "serde_json"))]
            content: MessageContent::Left("You are a helpful assistant.".to_string()),
            #[cfg(not(all(feature = "either", feature = "indexmap", feature = "serde_json")))]
            content: "You are a helpful assistant.".to_string(),
        }],
    };
    let serialized = serde_json::to_string(&task_input).unwrap();
    let deserialized: TaskInput = serde_json::from_str(&serialized).unwrap();
    assert_eq!(task_input, deserialized);
    // Примерный вид JSON: {"Chat":{"messages":[{"role":"system","content":"You are a helpful assistant."}]}}
     assert!(serialized.contains("Chat"));
}

#[test]
fn test_usage_metrics_serialization() {
    let metrics = UsageMetrics {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let serialized = serde_json::to_string(&metrics).unwrap();
    let deserialized: UsageMetrics = serde_json::from_str(&serialized).unwrap();
    assert_eq!(metrics, deserialized);
}

#[test]
fn test_inference_response_data_serialization() {
    let response = InferenceResponse::Data {
        request_id: "req-123".to_string(),
        text_chunk: "Hello".to_string(),
        is_final_chunk: false,
        usage: Some(UsageMetrics {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
        }),
    };
    let serialized = serde_json::to_string(&response).unwrap();
    let deserialized: InferenceResponse = serde_json::from_str(&serialized).unwrap();
    assert_eq!(response, deserialized);
    // Примерный вид JSON (из-за untagged): {"request_id":"req-123","text_chunk":"Hello","is_final_chunk":false,"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}
    assert!(serialized.contains("text_chunk"));
}

#[test]
fn test_inference_response_done_serialization() {
    let response = InferenceResponse::Done {
        request_id: "req-124".to_string(),
        full_text: "Hello world, this is the final response.".to_string(),
        usage: UsageMetrics {
            prompt_tokens: 10,
            completion_tokens: 50,
            total_tokens: 60,
        },
        stop_reason: "length".to_string(),
    };
    let serialized = serde_json::to_string(&response).unwrap();
    let deserialized: InferenceResponse = serde_json::from_str(&serialized).unwrap();
    assert_eq!(response, deserialized);
    // Примерный вид JSON (из-за untagged): {"request_id":"req-124","full_text":"...","usage":{...},"stop_reason":"length"}
    assert!(serialized.contains("full_text"));
}

#[test]
fn test_inference_response_error_serialization() {
    let response = InferenceResponse::Error {
        request_id: "req-125".to_string(),
        error: "Model not found".to_string(),
    };
    let serialized = serde_json::to_string(&response).unwrap();
    let deserialized: InferenceResponse = serde_json::from_str(&serialized).unwrap();
    assert_eq!(response, deserialized);
     // Примерный вид JSON (из-за untagged): {"request_id":"req-125","error":"Model not found"}
    assert!(serialized.contains("error"));
}

// Тест для InferenceTask потребует мокинг Sender, если with_tokio_sender включен.
// Пока оставим без него, так как он не реализует Serialize/Deserialize.
// Можно добавить тест на конструктор, если необходимо.
