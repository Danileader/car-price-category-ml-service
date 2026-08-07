# Car Price Category Prediction Service

Сервис для предсказания категории стоимости автомобиля по параметрам объявления.

Проект делался как практическая работа по полному циклу ML-задачи: подготовка данных, обучение модели, сохранение пайплайна, API для инференса и простой мониторинг нагрузки.

## Что внутри

- Обучение модели для многоклассовой классификации `price_category`.
- `sklearn` pipeline с обработкой пропусков, масштабированием числовых признаков и one-hot encoding категорий.
- Сравнение нескольких моделей через cross-validation.
- Сохранение обученного пайплайна через `joblib`.
- FastAPI endpoint для получения предсказаний.
- Scheduler, который каждые 10 секунд делает batch prediction на небольшой выборке.
- Демо-мониторинг через Grafana/MariaDB на искусственно сгенерированных метриках нагрузки.

## Project Structure

```text
car-price-category-ml-service/
  app/
    main.py                         # FastAPI application
  model/
    train.py                        # training script
    data/cars_prepared.csv          # prepared training data
    artifacts/cars_pipe.pkl         # trained model artifact
  monitoring/
    batch_predict_scheduler.py      # scheduled predictions every 10 seconds
    generate_load_metrics.py        # synthetic API/load metrics generator
    docker-compose.yml              # MariaDB, Adminer, Grafana
    init/grafana.sql                # metrics table seed
  examples/
    predict_payload.json            # example API request body
  Dockerfile
  requirements.txt
```

## Задача

Нужно по характеристикам автомобиля предсказать ценовую категорию объявления.

Используемые признаки:

- region
- year
- manufacturer
- model
- fuel
- odometer
- title status
- transmission
- state
- latitude and longitude

Целевая переменная: `price_category`.

## Модель

В `model/train.py` сравниваются несколько моделей:

- Logistic Regression
- Random Forest
- XGBoost, if installed

Лучшая модель по cross-validation обучается на подготовленном датасете и сохраняется в:

```text
model/artifacts/cars_pipe.pkl
```

Сохраненный артефакт содержит:

- trained preprocessing + model pipeline
- label encoder
- model metadata

## Запуск

Создать и активировать виртуальное окружение:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить API:

```bash
uvicorn app.main:app --reload
```

Документация FastAPI:

```text
http://127.0.0.1:8000/docs
```

## API

Проверка статуса:

```http
GET /status
```

Метаданные модели:

```http
GET /version
```

Предсказание:

```http
POST /predict
```

Пример запроса:

```json
{
  "region": "baltimore",
  "year": 2013,
  "manufacturer": "ford",
  "model": "mustang",
  "fuel": "gas",
  "odometer": 85000,
  "title_status": "clean",
  "transmission": "manual",
  "state": "md",
  "lat": 39.1618,
  "long": -76.6297
}
```

Пример ответа:

```json
{
  "price_category": "medium"
}
```

## Переобучение модели

```bash
python -m model.train
```

## Scheduler

Scheduler берет случайную часть подготовленного датасета и каждые 10 секунд выводит распределение предсказанных категорий:

```bash
python monitoring/batch_predict_scheduler.py
```

## Мониторинг

В папке `monitoring` лежит демо-стек для визуализации искусственных метрик нагрузки.

Запуск MariaDB, Adminer и Grafana:

```bash
cd monitoring
copy .env.example .env
docker compose up -d
```

Сервисы:

- Grafana: `http://localhost:3000`
- Adminer: `http://localhost:8080`
- MariaDB database: `metrics`

Таблица с метриками содержит:

- timestamp
- CPU usage
- available memory
- requests per minute
- median processing time

## Что можно улучшить

- Добавить train/test split и отдельный отчет по качеству на holdout.
- Добавить сохранение метрик обучения в отдельный файл.
- Логировать реальные запросы API и latency вместо полностью искусственных метрик.
- Добавить Docker Compose для API и мониторинга в одном окружении.
- Добавить тесты для endpoint `/predict`.

## Короткое описание для резюме

Разработал ML-сервис для предсказания категории стоимости автомобиля: подготовил данные, обучил и сериализовал `sklearn`/`XGBoost` pipeline, реализовал FastAPI endpoint для инференса, batch scheduler и демо-мониторинг нагрузки через Grafana/MariaDB.
