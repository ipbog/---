# scripts/auto-checklist.ps1
# Пример скрипта для автоматической проверки перед коммитом или релизом

Write-Host "Запуск автоматических проверок..."

# 1. Форматирование кода
Write-Host "Форматирование кода (cargo fmt)..."
cargo fmt --all -- --check
if ($LASTEXITCODE -ne 0) {
    Write-Error "Ошибка форматирования кода. Запустите 'cargo fmt --all'."
    # exit 1 # Раскомментируйте, чтобы прервать выполнение при ошибке
}

# 2. Линтинг (Clippy)
Write-Host "Линтинг кода (cargo clippy)..."
cargo clippy --all-targets --all-features -- -D warnings # Строгий режим: все предупреждения как ошибки
if ($LASTEXITCODE -ne 0) {
    Write-Error "Обнаружены ошибки Clippy."
    # exit 1
}

# 3. Сборка проекта (проверка компиляции)
Write-Host "Сборка проекта (cargo build)..."
cargo build --workspace --all-features
if ($LASTEXITCODE -ne 0) {
    Write-Error "Ошибка сборки проекта."
    # exit 1
}

# 4. Запуск тестов
Write-Host "Запуск тестов (cargo test)..."
cargo test --workspace --all-features
if ($LASTEXITCODE -ne 0) {
    Write-Error "Не все тесты пройдены."
    # exit 1
}

# 5. Проверка документации (опционально)
# Write-Host "Сборка документации (cargo doc)..."
# cargo doc --workspace --all-features --no-deps
# if ($LASTEXITCODE -ne 0) {
#     Write-Error "Ошибка сборки документации."
#     # exit 1
# }

Write-Host "Все автоматические проверки успешно пройдены!"
