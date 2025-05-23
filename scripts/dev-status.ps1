# scripts/dev-status.ps1
# Пример скрипта для отображения текущего статуса разработки

Write-Host "Статус разработки Model-Hub:"
Write-Host "--------------------------------"

# Количество TODO в коде (простой поиск)
$todo_count = (Get-ChildItem -Path .. -Recurse -Include *.rs | Select-String -Pattern "TODO").Count
Write-Host "Количество TODO в коде: $todo_count"

# Статус git (если бы использовался)
# if (Get-Command git -ErrorAction SilentlyContinue) {
#     Write-Host ""
#     Write-Host "Статус Git:"
#     git status --short
# }

# Информация о последнем коммите (если бы использовался)
# if (Get-Command git -ErrorAction SilentlyContinue) {
#     $last_commit = git log -1 --pretty=format:"%h - %s (%cr) <%an>"
#     Write-Host "Последний коммит: $last_commit"
# }

# Можно добавить вывод из tasks.md или bugs.md (потребует парсинга Markdown)
Write-Host ""
Write-Host "Проверьте tasks.md и bugs.md для детальной информации."
