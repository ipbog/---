// utils_crate/tests/sampling_params_tests.rs
#[cfg(feature = "inference_types_serde")]
mod sampling_params_serde_tests {
    use utils_crate::sampling_params::{SamplingParams, GenerationConstraint, StopTokens};
    use serde_json;
    use std::collections::HashMap;

    #[cfg(feature = "serde_json")] // Для GenerationConstraint::JsonSchema
    use serde_json::json;

    #[test]
    fn test_sampling_params_default_serialization() {
        let params = SamplingParams::default();
        let serialized = serde_json::to_string_pretty(&params).unwrap();
        let deserialized: SamplingParams = serde_json::from_str(&serialized).unwrap();
        assert_eq!(params, deserialized);
    }

    #[test]
    fn test_sampling_params_custom_serialization() {
        let params = SamplingParams {
            temperature: Some(0.5),
            top_k: Some(10),
            top_p: Some(0.95),
            repeat_penalty: Some(1.2),
            presence_penalty: Some(0.1),
            logits_bias: Some(HashMap::from([(101u32, 0.5f32), (102u32, -0.5f32)])),
            seed: Some(12345),
            constraint: GenerationConstraint::Regex("^\d+$".to_string()),
            ..Default::default() // min_p, top_n_logprobs будут из Default
        };
        let serialized = serde_json::to_string_pretty(&params).unwrap();
        let deserialized: SamplingParams = serde_json::from_str(&serialized).unwrap();
        assert_eq!(params, deserialized);
    }

    #[cfg(all(feature = "inference_types_serde", feature = "serde_json"))]
    #[test]
    fn test_generation_constraint_json_schema_serialization() {
        let constraint = GenerationConstraint::JsonSchema(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            }
        }));
        let serialized = serde_json::to_string(&constraint).unwrap();
        let deserialized: GenerationConstraint = serde_json::from_str(&serialized).unwrap();
        assert_eq!(constraint, deserialized);
    }

    #[test]
    fn test_stop_tokens_serialization() {
        let stop_seqs = StopTokens::Seqs(vec!["\nUser:".to_string(), "###".to_string()]);
        let serialized_seqs = serde_json::to_string(&stop_seqs).unwrap();
        let deserialized_seqs: StopTokens = serde_json::from_str(&serialized_seqs).unwrap();
        assert_eq!(stop_seqs, deserialized_seqs);

        let stop_ids = StopTokens::Ids(vec![50256, 1, 2]);
        let serialized_ids = serde_json::to_string(&stop_ids).unwrap();
        let deserialized_ids: StopTokens = serde_json::from_str(&serialized_ids).unwrap();
        assert_eq!(stop_ids, deserialized_ids);

        let stop_none = StopTokens::None;
        let serialized_none = serde_json::to_string(&stop_none).unwrap();
        let deserialized_none: StopTokens = serde_json::from_str(&serialized_none).unwrap();
        assert_eq!(stop_none, deserialized_none);
    }

    #[test]
    fn test_sampling_params_deterministic() {
        let params = SamplingParams::deterministic();
        assert_eq!(params.temperature, Some(0.0));
        assert_eq!(params.top_k, Some(1));
        assert_eq!(params.top_p, None); // Убедимся, что top_p отключается
                                       // Проверяем, что остальные параметры не изменились неожиданно
        let default_params = SamplingParams::default();
        assert_eq!(params.min_p, default_params.min_p);
        assert_eq!(params.repeat_penalty, default_params.repeat_penalty);
        assert_eq!(params.presence_penalty, default_params.presence_penalty);
        assert_eq!(params.logits_bias, default_params.logits_bias);
        assert_eq!(params.top_n_logprobs, default_params.top_n_logprobs);
        assert_eq!(params.seed, default_params.seed); // seed сохраняется из default
        assert_eq!(params.constraint, default_params.constraint);
    }
}
