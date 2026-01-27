# API Contracts (MVP)

## Authentication

This repo supports **two** auth mechanisms:

1) **End-user JWT** (recommended):
- Obtain token from `auth_service`:
  - `POST /v1/auth/token` (username/password form)
- Use:
  - `Authorization: Bearer <token>`

2) **Internal service API key** (service-to-service only):
- `X-API-Key: <API_KEY>`

In “secure mode” (`docker-compose.yml`), only the **gateway** is exposed to the host; internal services are reachable only inside the docker network.

---

## Auth Service – `http://auth:8004`

### POST `/v1/auth/token`
Form fields:
- `username`
- `password`

Returns:
- `access_token` (JWT)
- `expires_in`

### GET `/v1/auth/me`
Headers:
- `Authorization: Bearer ...`

---

## Preprocess Service (Stage 1) – `http://preprocess:8001`

### POST `/v1/preprocess`
Headers:
- `Authorization: Bearer ...`  (or internal `X-API-Key`)

Multipart form:
- `file`: PDF/image
- `patient_id` (optional)
- `document_date` (optional, ISO)

Returns:
- `entities` (NER best-effort)
- `events` (best-effort event hints)
- `text` may be blank if `RETURN_OCR_TEXT=false`

---

## Patient Store Service (Stage 2) – `http://patient-store:8002`

All endpoints require:
- `Authorization: Bearer ...`  (or internal `X-API-Key`)

### POST `/v1/patients`
Create a patient record.

### GET `/v1/patients`
Lists patients (demo only; lock down/limit in real deployments)

### POST `/v1/patients/{patient_id}/events:bulk`
Store structured events.

### GET `/v1/patients/{patient_id}/events`
Query params:
- `from` (optional ISO)
- `to` (optional ISO)

### POST `/v1/patients/{patient_id}/documents`
Multipart form:
- `file`
- `extraction_json` (stringified JSON returned from preprocess)

---

## Inference Service (Stage 3) – `http://inference:8003`

### POST `/v1/predict`
Headers:
- `Authorization: Bearer ...`  (or internal `X-API-Key`)

JSON:
- Either:
  - `patient_id` (preferred), OR
  - `events` (direct event list)
- `as_of` (optional ISO date-time)
- `top_n` (optional)
- `return_debug` (optional)

Returns:
- `predictions`: predicted condition codes w/ probabilities and recommendations
- `threshold` used
- optional `debug`
