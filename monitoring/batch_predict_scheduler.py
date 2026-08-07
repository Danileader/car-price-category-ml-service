from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import tzlocal
from apscheduler.schedulers.blocking import BlockingScheduler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "model" / "data" / "cars_prepared.csv"
MODEL_PATH = ROOT_DIR / "model" / "artifacts" / "cars_pipe.pkl"

scheduler = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv(DATA_PATH)
features = df.drop(columns=["price_category"], errors="ignore")

loaded = joblib.load(MODEL_PATH)
model = loaded["model"]
label_encoder = loaded["label_encoder"]


@scheduler.scheduled_job("cron", second="*/10")
def predict_sample() -> None:
    sample = features.sample(frac=0.05, random_state=None)
    preds_encoded = model.predict(sample)
    preds_decoded = label_encoder.inverse_transform(preds_encoded)

    stats = pd.Series(preds_decoded).value_counts(normalize=True).mul(100).round(1)

    print(f"\n{datetime.now().strftime('%H:%M:%S')}")
    print(f"Predicted rows: {len(sample)}")
    print("Predicted category distribution:")
    for category, percent in stats.items():
        print(f"  {category}: {percent}%")


if __name__ == "__main__":
    print("Starting batch prediction scheduler. Press Ctrl+C to stop.")
    scheduler.start()

