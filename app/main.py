from pathlib import Path
import argparse
import logging
import time

from dotenv import load_dotenv

from app.backend_client import MiniweatherBackendClient
from app.config import load_settings
from app.model_runner import LSTMAutoencoderRunner
from app.scylla_client import ScyllaWeatherClient
from app.service import run_once


def build_runners() -> list[LSTMAutoencoderRunner]:
    root = Path(__file__).resolve().parents[1]
    return [
        LSTMAutoencoderRunner(root / "artifacts" / "deployment" / "rain_7d"),
        LSTMAutoencoderRunner(root / "artifacts" / "deployment" / "wind_30d"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["once", "worker"], help="Run once or loop forever")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    settings = load_settings()
    weather_client = ScyllaWeatherClient(settings)
    backend_client = MiniweatherBackendClient(settings)
    runners = build_runners()

    try:
        if args.mode == "once":
            run_once(settings, weather_client, backend_client, runners)
            return

        while True:
            run_once(settings, weather_client, backend_client, runners)
            time.sleep(settings.run_interval_minutes * 60)
    finally:
        weather_client.close()


if __name__ == "__main__":
    main()
