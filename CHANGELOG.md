# Журнал Изменений (Changelog)

Все заметные изменения в этом проекте будут документированы в этом файле.
Формат основан на [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
и проект придерживается [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Не выпущено]

### Добавлено
- Начальная структура проекта с крейтами: `utils_crate`, `core_burn`, `model_loader`, `inference_engine`, `lsp_server_backend`, `gui_app`, `cli_app`.
- Базовая конфигурация `Cargo.toml` для всего воркспейса и для каждого крейта.
- Определение API и структур для `utils_crate`, `core_burn` (включая `KvCache`), `model_loader`, `inference_engine`.
- Заглушка для `InferenceTokenizer` в `inference_engine`.
- Основа для `gui_app` на `egui` с `LspClient` для взаимодействия с будущим LSP сервером.
- Скелет для `lsp_server_backend` для обработки JSON-RPC и базовых LSP запросов.
- Шаблоны для мета-файлов (`README.md`, `CHANGELOG.md`, `documentation.md`, `bugs.md`, `tasks.md`).
- Настройка `rust-toolchain.toml`.
- Начальные скрипты в `scripts/`.

### Изменено
- Перемещена реализация `KvCache` из `inference_engine` в `core_burn`.
- `model_loader` теперь использует `GemmaModelConfig` из `core_burn`.
- Уточнены структуры `InferenceTask` и `SamplerParams` в `utils_crate`.
- `InferenceConfig` в `inference_engine` переименован в `EngineConfig` и упрощен.
- Обновлены версии зависимостей в `Cargo.toml` файлах.

### Исправлено
- Устранена проблема с фичей `derive` для зависимости `burn` во всех `Cargo.toml`.

### Удалено
- Файл `burn_safetensors_loader.rs` из `core_burn` (логика перенесена в `model_loader`).

## [0.1.0] - YYYY-MM-DD (Планируемый первый релиз)

*   ... список будущих изменений для версии 0.1.0 ...
```
