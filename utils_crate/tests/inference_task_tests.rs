// utils_crate/tests/inference_task_tests.rs
#[cfg(feature = "inference_types_serde")]
mod inference_task_serde_tests {
    use utils_crate::inference_task::{
        ChatMessage, InferenceResponse, TaskInput, UsageMetrics, MessageContent,
    };
    use serde_json;
    #[cfg(all(feature = "either", feature = "indexmap"))] // Для сложного MessageContent
    use {either::Either, indexmap::indexmap};


    #[test]
    fn test_chat_message_serialization_deserialization_simple_content() {
        let original_msg = ChatMessage {
            role: "user".to_string(),
            content: "Привет, мир!".to_string(), // Используем String, так как MessageContent может быть String
        };

        let serialized = serde_json::to_string(&original_msg).unwrap();
        let deserialized: ChatMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original_msg, deserialized);
    }

    #[cfg(all(feature = "either", feature = "indexmap", feature = "serde_json"))]
    #[test]
    fn test_chat_message_serialization_deserialization_complex_content() {
        let complex_content_vec = vec![
            indexmap! { "type".to_string() => serde_json::Value::String("text".to_string()), "text".to_string() => serde_json::Value::String("Это текстовая часть.".to_string()) },
            indexmap! { "type".to_string() => serde_json::Value::String("image_url".to_string()), "image_url".to_string() => serde_json::json!({ "url": "http://example.com/image.png" }) },
        ];

        let original_msg = ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Right(complex_content_vec.clone()),
        };

        let serialized = serde_json::to_string(&original_msg).unwrap();
        println!("Serialized complex: {}", serialized);
        let deserialized: ChatMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original_msg.role, deserialized.role);
        match (original_msg.content, deserialized.content) {
            (MessageContent::Right(orig_vec), MessageContent::Right(deser_vec)) => {
                assert_eq!(orig_vec.len(), deser_vec.len());
                for (o, d) in orig_vec.iter().zip(deser_vec.iter()) {
                    assert_eq!(o, d);
                }
            }
            _ => panic!("Неверный тип MessageContent после десериализации сложного контента"),
        }
    }


    #[test]
    fn test_task_input_completion_serialization() {
        let input = TaskInput::Completion { text: "Заверши это: ".to_string(), echo_prompt: true };
        let serialized = serde_json::to_string(&input).unwrap();
        let deserialized: TaskInput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(input, deserialized);
    }

    #[test]
    fn test_task_input_chat_serialization() {
        let input = TaskInput::Chat { messages: vec![
            ChatMessage { role: "system".to_string(), content: "Ты полезный ассистент.".to_string() },
            ChatMessage { role: "user".to_string(), content: "Как дела?".to_string() },
        ]};
        let serialized = serde_json::to_string(&input).unwrap();
        let deserialized: TaskInput = serde_json::from_str(&serialized).unwrap();
        assert_eq!(input, deserialized);
    }

    #[test]
    fn test_inference_response_data_serialization() {
        let response = InferenceResponse::Data {
            request_id: "req-123".to_string(),
            text_chunk: "Это часть ".to_string(),
            is_final_chunk: false,
            usage: Some(UsageMetrics { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }),
        };
        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: InferenceResponse = serde_json::from_str(&serialized).unwrap();
        assert_eq!(response, deserialized);
    }

    #[test]
    fn test_inference_response_done_serialization() {
        let response = InferenceResponse::Done {
            request_id: "req-124".to_string(),
            full_text: "Это полный ответ.".to_string(),
            usage: UsageMetrics { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
            stop_reason: "length".to_string(),
        };
        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: InferenceResponse = serde_json::from_str(&serialized).unwrap();
        assert_eq!(response, deserialized);
    }

    #[test]
    fn test_inference_response_error_serialization() {
        let response = InferenceResponse::Error {
            request_id: "req-125".to_string(),
            error: "Произошла ошибка модели.".to_string(),
        };
        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: InferenceResponse = serde_json::from_str(&serialized).unwrap();
        assert_eq!(response, deserialized);
    }
}

// Тесты для InferenceTask без tokio::sync::mpsc::Sender, так как он не Serialize/Deserialize
// и не Debug (если не обернут).
// Если InferenceTask должен быть сериализуем, response_sender нужно будет обработать
// с помощью #[serde(skip)] или подобного.
