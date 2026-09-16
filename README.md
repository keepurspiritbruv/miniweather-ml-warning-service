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

## 3. Konfigurasi (`.env`)

Salin `.env.example` menjadi `.env` lalu isi:

| Variabel | Default | Keterangan |
| -------- | ------- | ---------- |
| `DATA_SOURCE` | `scylla` | sumber data |
| `SCYLLA_CONTACT_POINTS` | `10.42.28.70` | IP host Scylla |
| `SCYLLA_PORT` | `9042` | port CQL |
| `SCYLLA_KEYSPACE` | `hyperbase` | keyspace |
| `SCYLLA_USERNAME` / `SCYLLA_PASSWORD` | — | kredensial Scylla (wajib) |
| `SCYLLA_TABLE` | `records_91e2e000cb17494c8a45c6182b2a89ac` | tabel record device |
| `SCYLLA_COLLECTION_ID` | `91e2e000-cb17-494c-8a45-c6182b2a89ac` | di-bind sebagai `uuid` |
| `SCYLLA_QUERY_LIMIT` | `250000` | batas baris per query |
| `MINIWEATHER_API_BASE_URL` | `http://miniweather-backend:3001` | base URL backend |
| `MINIWEATHER_AUTH_EMAIL` / `MINIWEATHER_AUTH_PASSWORD` | — | akun **admin/superAdmin** backend (wajib) |
| `RUN_INTERVAL_MINUTES` | `30` | interval worker |
| `TIMEZONE` | `Asia/Jakarta` | timezone agregasi harian |
| `DRY_RUN` | `true` | bila `true`, tidak mengirim warning |
| `SKIP_PARTIAL_TODAY` | `true` | scoring hanya sampai hari lengkap terakhir |
| `ALERT_COOLDOWN_HOURS` | `12` | jeda minimum warning level sama per hazard |
| `ALERT_STATE_FILE` | `alert_state.json` | file state cooldown |
| `STALE_AFTER_HOURS` | `48` | ambang data dianggap basi |
| `RAIN_SIAGA_MM` / `RAIN_AWAS_MM` | `50` / `100` | threshold fisik hujan |
| `WIND_SIAGA_MS` / `WIND_AWAS_MS` | `10.8` / `17.2` | threshold fisik angin |

---

## 4. Aturan Alert

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

---

## 5. Integrasi Backend

| Aksi | Endpoint |
| ---- | -------- |
| Login admin | `POST {BASE}/v1/auth/login` → ambil `data.accessToken` |
| Kirim warning | `POST {BASE}/v1/warnings` (header `Authorization: Bearer <token>`) |
| Body warning | `{ "message", "type": "weather", "is_active": true, "source": "ml", "hazard": "<hazard>", "level": "<LEVEL>" }` |

`source`/`hazard`/`level` didukung backend (kolom terstruktur). Bila backend lama belum punya kolom tersebut, field tambahan diabaikan (tidak error). Akun pada `.env` harus ber-role `admin`/`superAdmin` dan `is_active=true`.

---

## 6. Akses ScyllaDB

- Tabel record: `records_91e2e000cb17494c8a45c6182b2a89ac` (PK `("_collection_id","_id")`).
- `_collection_id` bertipe `uuid` → service mem-parse ke `uuid.UUID` sebelum bind.
- Query dijalankan sebagai **prepared statement**; hasil dibaca via nama kolom driver (`row.id`, `row.collection_id`, `row.updated_at`, dst).
- Agregasi harian (timezone `Asia/Jakarta`): `RR = sum(curah_hujan)`, `ff_x = max(kecepatan_angin)`, `ff_avg = mean(kecepatan_angin)`, `arah_angin = mean(arah_angin)`.

> Perlu dikonfirmasi admin: apakah `curah_hujan` IoT bersifat incremental (per record) atau cumulative. Jika cumulative, agregasi `sum` harus diganti.

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

## 10. Troubleshooting

- **`TypeError` bind UUID** → pastikan `SCYLLA_COLLECTION_ID` valid UUID (sudah di-handle kode).
- **`not all arguments converted`** → jalur query harus via prepared statement (sudah diterapkan).
- **`AttributeError` saat baca row** → akses kolom memakai nama tanpa underscore (`row.id`, bukan `row._id`).
- **`Unrecognized name updated_at`** → query memakai `_updated_at` (kolom Scylla), bukan `updated_at`.
- **Scylla timeout** → cek `SCYLLA_CONTACT_POINTS`/port & firewall.

---

## 11. Keamanan

- Jangan commit `.env`.
- Gunakan akun admin khusus untuk service bila memungkinkan.
- `DRY_RUN=true` sampai hasil dry-run tervalidasi.
