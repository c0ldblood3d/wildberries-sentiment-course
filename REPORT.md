# Отчёт о курсовом проекте

Тема: воспроизводимый сервис классификации тональности русскоязычных отзывов Wildberries.

| Материал | Ссылка или идентификатор |
|---|---|
| Репозиторий | [Публичный GitHub](https://github.com/c0ldblood3d/wildberries-sentiment-course) |
| Итоговый tag | [v1.0.0](https://github.com/c0ldblood3d/wildberries-sentiment-course/tree/v1.0.0); полный SHA: `git ls-remote origin refs/tags/v1.0.0` |
| Код исходного обучения | Commit `83c2a6d772d9d9577521a1863b320974dc92ba47`, архив `training-source.zip` в релизе |
| CI | [Успешный запуск базового проекта](https://github.com/c0ldblood3d/wildberries-sentiment-course/actions/runs/44086499707); [все проверки, включая итоговую версию](https://github.com/c0ldblood3d/wildberries-sentiment-course/actions) |
| Итоговая модель | [final_model.joblib](https://github.com/c0ldblood3d/wildberries-sentiment-course/releases/download/v1.0.0/final_model.joblib) |
| Эксперименты | [experiment-evidence.zip](https://github.com/c0ldblood3d/wildberries-sentiment-course/releases/download/v1.0.0/experiment-evidence.zip); MLflow experiment ID `1` |
| Лучший validation run | `dc94e7f4625c45ed93bb953c01e143d1` |
| Итоговый artifact run | `f92bdf85627149dfac8f5f72fd2ec739` |
| Manifest данных и модели | [release-manifest.json](https://github.com/c0ldblood3d/wildberries-sentiment-course/releases/download/v1.0.0/release-manifest.json) |
| Демо | Ожидается запись студентом по `DEMO_SCRIPT.md` (1–3 минуты) |

## Результаты

Сравнение выполнено на едином групповом разбиении по `product_id` (примерно 70/10/20). Критерий выбора — validation macro F1.

| Модель | Validation macro F1 |
|---|---:|
| Word TF-IDF + Logistic Regression, C=1 | 0.6434 |
| Word TF-IDF + Logistic Regression, C=3 | **0.6536** |
| Word + character TF-IDF + LinearSVC | 0.6274 |

После выбора модель обучена на train+validation и однократно оценена на test: macro F1 **0.6276**, accuracy **0.8194**, weighted F1 **0.8247**. Это классификация меток по рейтингу: она не равнозначна ручной разметке эмоций.

## Данные

Исходник передан студентом; публичная лицензия на распространение не подтверждена. CSV предоставляется преподавателю отдельно. Воспроизведение обучения требует именно исходного файла `filtered.csv` с SHA-256 `eca8620637127841f3f8cc108ffbee839e18d0584de79ed5eef74c8e020b9cf6`.

- v1 SHA-256: `4d6b3e22b29373b1bd49bdc5c792ba620d68ff8c6ef21ddb24cd2d9f46842ef4`.
- v2 SHA-256: `76ea7497edbf65a48359b77aea27cc30f3b99e76a9eb6e786b67aa43225b2672`.
- Model SHA-256: `4d661f83494874ac25e8e5aa2403b7206eb5479395a2bd0648e3071b61116e4d`.

Архив экспериментов содержит SQLite store, параметры, метрики, модельные artifacts, `reports/experiments.csv`, `reports/test_report.json` и manifest версий данных. Распаковать в отдельную папку, выполнить `python relocate.py`, затем запустить `mlflow ui --backend-store-uri sqlite:///./mlflow.db --host 127.0.0.1` из окружения проекта. В интерфейсе выбрать experiment `1` и run по указанному ID.

## Проверка сдачи

Клонировать tag, восстановить окружение по lock-файлу, скачать модель в `artifacts/`, проверить её SHA-256, запустить тесты и Docker smoke по README. Для повторения обучения отдельно получить исходный CSV. Структура проекта, конфигурация, тесты и модель описаны в README и карточках. Видеоссылка будет добавлена после записи.
