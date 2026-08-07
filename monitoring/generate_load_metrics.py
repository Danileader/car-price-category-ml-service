import datetime
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

import psutil
import requests
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000/predict"
ITERATIONS = 720


def load_cpu(duration: int) -> None:
    start_time = time.time()
    while time.time() - start_time <= duration:
        _ = pow(random.randint(1, 1000), 32)


def generate_request_payload() -> dict:
    return {
        "region": "baltimore",
        "year": 2013,
        "manufacturer": "ford",
        "model": "mustang",
        "fuel": "gas",
        "odometer": random.randint(1000, 500000),
        "title_status": "clean",
        "transmission": "manual",
        "state": "md",
        "lat": 39.1618,
        "long": -76.6297,
    }


def write_sql_header(file) -> None:
    header = """
CREATE DATABASE IF NOT EXISTS metrics;
USE metrics;

DROP TABLE IF EXISTS metrics;

CREATE TABLE metrics (
    timestamp DATETIME NOT NULL,
    cpu_usage DECIMAL(5, 2) NOT NULL,
    mem_available BIGINT NOT NULL,
    reqs_per_min INT NOT NULL,
    time_of_proc DECIMAL(5, 2) NOT NULL,
    PRIMARY KEY (timestamp)
) ENGINE=InnoDB;

INSERT INTO metrics (timestamp, cpu_usage, mem_available, reqs_per_min, time_of_proc) VALUES
"""
    file.write(header.strip() + "\n")


def collect_metrics(cpu_pool: ProcessPoolExecutor, current_time: datetime.datetime):
    start_time = time.time()
    latencies = []
    request_count = 0

    while True:
        request_start = time.time()

        try:
            requests.post(API_URL, json=generate_request_payload(), timeout=1)
        except requests.RequestException:
            pass
        else:
            request_count += 1
            latencies.append(time.time() - request_start)

        if time.time() - start_time > 1:
            break

    if random.randint(0, 1):
        cpu_pool.submit(load_cpu, random.randint(1, 10))

    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().available
    latency_ms = median(latencies) * 1000 if latencies else 0

    return current_time, cpu, memory, request_count, latency_ms


def main(output_path: str) -> None:
    cpu_pool = ProcessPoolExecutor()
    current_time = datetime.datetime.utcnow() - datetime.timedelta(hours=10)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        write_sql_header(file)

        for i in tqdm(range(ITERATIONS), desc="Generating metrics"):
            metrics = collect_metrics(cpu_pool, current_time)

            if i > 0:
                file.write(",\n")

            file.write(
                f'("{metrics[0]}", {metrics[1]}, {metrics[2]}, '
                f"{metrics[3]}, {metrics[4]:.2f})"
            )
            file.flush()
            current_time += datetime.timedelta(minutes=1)

        file.write(";\n")


if __name__ == "__main__":
    main(str(ROOT_DIR / "monitoring" / "init" / "grafana.sql"))

