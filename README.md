# Miniweather ML Warning Service

Worker Python untuk deteksi anomali cuaca DI Yogyakarta dengan **LSTM Autoencoder**, dipakai oleh Miniweather Station Dashboard. Service membaca data IoT langsung dari ScyllaDB, membangun fitur harian, menjalankan dua model deployment (1 model per bencana), lalu mengirim warning ke backend Miniweather.

- **Hujan ekstrem** → model `rain_7d` (`curah_hujan_tinggi`)
- **Angin kencang** → model `wind_30d` (`angin_kencang`)

Kode riset, notebook, dan hasil analisis model ada di repo terpisah yang dipasang sebagai **git submodule**: folder [`AnomalyDetectionLSTM/`](https://github.com/keepurspiritbruv/AnomalyDetectionLSTM).

---

## 1. Arsitektur & Alur

```text
IoT device  --MQTT-->  backend Miniweather  --write-->  ScyllaDB (keyspace: hyperbase)
                                                             |
                                                             |  (baca langsung)
                                                             v
                          miniweather-ml-warning-service (repo ini)
                          - ambil 35 hari terakhir
                          - fitur harian (RR, rain_3d, rain_7d, ff_x, ...)
                          - scoring LSTM Autoencoder (rain_7d, wind_30d)
                          - rule status: NORMAL / WASPADA / SIAGA / AWAS
                                                             |
                                       POST /v1/warnings (Bearer token admin)
                                                             v
                                             backend Miniweather -> Postgres `warnings`
                                                             |
                                                             v
                                            frontend / adminpage menampilkan warning
```

Model tidak dilatih di service ini — hanya **inference**. Training dilakukan di repo submodule.

---

## 2. Model (Submodule)

| Model | Hazard | Window | Fitur | Artifact |
| ----- | ------ | -----: | ----- | -------- |
| `rain_7d` | `curah_hujan_tinggi` | 7 hari | `RR`, `rain_3d`, `rain_7d`, `rain_change_1d`, `missing_RR` | `artifacts/deployment/rain_7d/` |
| `wind_30d` | `angin_kencang` | 30 hari | `ff_x`, `ff_avg`, `wind_change_1d`, `ddd_x_sin`, `ddd_x_cos`, `missing_ff_x`, `missing_ff_avg` | `artifacts/deployment/wind_30d/` |

Setiap folder artifact berisi `model.keras`, `scaler.pkl`, `threshold.json` (P95/P99/P99.5), `feature_config.json`, dan `training_report.json`.

Clone beserta submodule:

```bash
git clone --recurse-submodules https://github.com/Miniweather-Station-Dashboard/miniweather-ml-warning-service.git
# atau bila sudah clone:
git submodule update --init --recursive
```

Detail model, sumber data, dan hasil analisis: lihat `AnomalyDetectionLSTM/README.md`.

---

## 3. Aturan Alert

| Level | Rule |
| ----- | ---- |
| `NORMAL` | anomaly score < P95 |
| `WASPADA` | anomaly score ≥ P95 (dari skor saja) |
| `SIAGA hujan` | score ≥ P99 **dan** curah hujan harian ≥ `RAIN_SIAGA_MM` |
| `AWAS hujan` | score ≥ P99.5 **dan** curah hujan harian ≥ `RAIN_AWAS_MM` |
| `SIAGA angin` | score ≥ P99 **dan** kecepatan angin maksimum ≥ `WIND_SIAGA_MS` |
| `AWAS angin` | score ≥ P99.5 **dan** kecepatan angin maksimum ≥ `WIND_AWAS_MS` |

Catatan:

- `SIAGA`/`AWAS` hanya aktif bila syarat fisik terpenuhi; jika tidak, status tetap `WASPADA`. Tidak ada eskalasi berbasis nilai fisik saja.
- `ALERT_COOLDOWN_HOURS`: level sama tidak dikirim ulang sebelum jeda; eskalasi selalu dikirim; penurunan level tidak dikirim. Saat kembali `NORMAL`, state cooldown di-reset.
- `SKIP_PARTIAL_TODAY=true`: hari berjalan (parsial) tidak discoring.
- Masked sliding window (aktif untuk artifact v2 yang punya `mask_policy`): hari/nilai hilang ditandai `mask=0` dan **tidak** dihitung di reconstruction error; gap pendek diimputasi untuk input (batas `IMPUTE_MAX_GAP_DAYS`, default `3`). Bila rasio observed di window `< MIN_OBSERVED_RATIO` (default `0.7`) → status `INSUFFICIENT_DATA` (tidak mengirim warning).

---

## 4. Integrasi Backend

| Aksi | Endpoint |
| ---- | -------- |
| Login admin | `POST {BASE}/v1/auth/login` → ambil `data.accessToken` |
| Kirim warning | `POST {BASE}/v1/warnings` (header `Authorization: Bearer <token>`) |
| Body warning | `{ "message", "type": "weather", "is_active": true, "source": "ml", "hazard": "<hazard>", "level": "<LEVEL>" }` |

`source`/`hazard`/`level` didukung backend (kolom terstruktur). Bila backend lama belum punya kolom tersebut, field tambahan diabaikan (tidak error). Akun pada `.env` harus ber-role `admin`/`superAdmin` dan `is_active=true`.

---

## 7. Menjalankan Service (Docker)

```bash
docker network create miniweather-net                # bila belum ada
docker network connect miniweather-net miniweather-backend

cp .env.example .env
chmod 600 .env
nano .env

docker build -t miniweather-ml-warning-service:latest .

# uji sekali (dry-run)
docker run --rm --network host --env-file .env \
  miniweather-ml-warning-service:latest python -m app.main once

# jalankan worker
docker run -d --name ml-warning-service \
  --restart unless-stopped --network host --env-file .env \
  -v "$PWD/alert_state.json:/app/alert_state.json" \
  miniweather-ml-warning-service:latest

docker logs -f ml-warning-service
```

Catatan:

- Service tidak membuka port (worker murni); `--network host` cukup untuk menjangkau Scylla & backend.
- Mount `alert_state.json` agar state cooldown bertahan saat container dibuat ulang.
- `DRY_RUN=true` sampai log divalidasi dan disetujui; baru ubah ke `false`.

---

## 8. Halaman Log

| Log status | Arti |
| ---------- | ---- |
| `SCORED` | model berjalan; baca status & score |
| `NORMAL` | tidak ada anomali |
| `WAITING_FOR_HISTORY` | window belum cukup / hari tidak kontinu |
| `WAITING_FOR_RECENT_HISTORY` | data IoT basi (> `STALE_AFTER_HOURS`) |
| `warning suppressed ... cooldown_active` | ditahan oleh cooldown |

---

## 9. Tes

```bash
python -m pytest -q tests
```

Tes berjalan tanpa TensorFlow/Scylla (dependency eksternal di-import secara lazy).

---

## 10. Training Artifact Masked v2 (Fase 2)

Skrip `scripts/train_deployment_v2.py` melatih ulang model deployment dengan **masked
sliding window** (hari hilang di-mask, tidak dihitung di error) dan mengekspor artifact
v2 (`threshold.json` berisi `total` + `per_feature`, `feature_config.json` memuat
`mask_policy`). Skrip memakai `app.masking` agar aturan mask training = inference.

Jalankan di environment yang punya TensorFlow (Colab/Jupyter), setelah notebook 01
menghasilkan `data/processed/yogyakarta_weather_features.csv`:

```bash
pip install -r requirements.txt          # berisi tensorflow
python scripts/train_deployment_v2.py \
  --processed data/processed/yogyakarta_weather_features.csv \
  --out artifacts/deployment_v2
```

Setelah artifact v2 tervalidasi (bandingkan dengan v1 lewat dry-run), promosikan ke v2:

```bash
# backup v1 lebih dulu, lalu salin v2
cp -r artifacts/deployment artifacts/deployment_v1_backup
cp -r artifacts/deployment_v2/* artifacts/deployment/
```

Service otomatis memakai jalur masked scoring begitu `feature_config.json` artifact
memuat `mask_policy` (lihat bagian 3).

---

## 11. Troubleshooting

- **`TypeError` bind UUID** → pastikan `SCYLLA_COLLECTION_ID` valid UUID (sudah di-handle kode).
- **`not all arguments converted`** → jalur query harus via prepared statement (sudah diterapkan).
- **`AttributeError` saat baca row** → akses kolom memakai nama tanpa underscore (`row.id`, bukan `row._id`).
- **`Unrecognized name updated_at`** → query memakai `_updated_at` (kolom Scylla), bukan `updated_at`.
- **Scylla timeout** → cek `SCYLLA_CONTACT_POINTS`/port & firewall.
