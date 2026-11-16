# Отчёт о курсовом проекте

## Тема

Воспроизводимый сервис классификации тональности русскоязычных отзывов Wildberries.

## Локальная проверка

Запустить `poetry install --sync`, подготовку данных, обучение, тесты и Docker smoke test по командам из README. Итоги обучения сохраняются в `reports/`, `artifacts/` и `mlruns/` и исключены из Git.

## Идентификаторы и ссылки для преподавателя

| Материал | Значение |
|---|---|
| Репозиторий | Будет опубликован как публичный GitHub-репозиторий |
| Итоговый tag | `v1.0.0` (локальный) |
| Commit SHA модели и эксперимента | `83c2a6d772d9d9577521a1863b320974dc92ba47`; SHA commit, на который указывает tag, получить командой `git rev-parse v1.0.0^{commit}` |
| CI pipeline | Workflow готов; ссылка появится после публикации GitHub-репозитория |
| MLflow experiment / runs | Локальная SQLite база, experiment ID `1`; winner run `dc94e7f4625c45ed93bb953c01e143d1`; final artifact run `f92bdf85627149dfac8f5f72fd2ec739` |
| Dataset v1/v2 | v1 SHA-256 `4d6b3e22b29373ddb24bd9c5c792ba620d68ff8c6ef21ddb24cd2d9f46842ef4`; v2 SHA-256 `76ea7497edbf65a48359b77aea27cc30f3b99e76a9eb6e786b67aa43225b2672` |
| Model artifact | Локальный `artifacts/final_model.joblib`; приложить к GitHub Release `v1.0.0` |
| Docker image digest (локально) | `sha256:96de5d85a370d19281f952b4a31fbd4e80316d5ef4142ca2405dd359ac0cd33` |
| Демо 1–3 минуты | Записать по сценарию `DEMO_SCRIPT.md` и добавить ссылку |

## Сценарий демо

1. Показать сборку Docker image и запуск.
2. Открыть `/health`.
3. Отправить корректный `/predict`.
4. Отправить пустой `text` и показать HTTP 422.
5. Показать версию модели и ссылку на итоговый эксперимент.

## Проверка соответствия исходному tag

После публикации указать полный commit SHA, выполнить проверку на чистом клоне и убедиться, что README, модельный artifact, данные, pipeline и видео соответствуют этому SHA.
