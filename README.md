# Miniweather ML Warning Service

Python worker service for Miniweather LSTM Autoencoder anomaly detection in Yogyakarta.

## Runtime Flow

```text
ScyllaDB records table
-> daily weather features
-> rain_7d and wind_30d LSTM Autoencoder models
-> alert rules
-> dry-run logs or POST /v1/warnings
```

## Setup On VPS

Create Docker network if it does not exist:

```bash
docker network create miniweather-net
```

Connect backend container:

```bash
docker network connect miniweather-net miniweather-backend
```

Prepare env:

```bash
cp .env.example .env
nano .env
```

Build:

```bash
docker build -t miniweather-ml-warning-service .
```

Run one dry-run check:

```bash
docker run --rm --network miniweather-net --env-file .env miniweather-ml-warning-service python -m app.main once
```

Run worker:

```bash
docker run -d \
  --name miniweather-ml-warning-service \
  --restart always \
  --network miniweather-net \
  --env-file .env \
  miniweather-ml-warning-service
```

Check logs:

```bash
docker logs -f miniweather-ml-warning-service
```

## Configuration

The service reads ScyllaDB directly from:

```text
keyspace: hyperbase
table: records_91e2e000cb17494c8a45c6182b2a89ac
collection id: 91e2e000-cb17-494c-8a45-c6182b2a89ac
```

It posts warning records to the Miniweather backend only when `DRY_RUN=false`.

Additional behavior:

- `SKIP_PARTIAL_TODAY=true` (default) scores only up to the last completed
  local day, because a still-running day has partial rainfall totals. Note this
  also means staleness (`STALE_AFTER_HOURS`) is measured from the last completed
  day, so keep the threshold above ~48h.
- `ALERT_COOLDOWN_HOURS` (default `12`) is the minimum interval between two
  warnings of the same level for the same hazard. Escalations are always sent;
  downgrades are never sent. When conditions return to NORMAL no "clear" warning
  is posted, so deactivate a still-active warning through the backend admin.
  Cooldown state is persisted in `ALERT_STATE_FILE` (default `alert_state.json`).
  Because that path lives on the container's writable layer, mount it as a volume
  (for example `-v /opt/miniweather/alert_state.json:/app/alert_state.json`) so
  cooldown state survives container recreation.
- The service refuses to score when the most recent window contains gaps
  (non-consecutive calendar days); it reports `WAITING_FOR_HISTORY` instead.

## Safety

Keep `DRY_RUN=true` until logs are reviewed by the lecturer/admin.

Never commit `.env`.
