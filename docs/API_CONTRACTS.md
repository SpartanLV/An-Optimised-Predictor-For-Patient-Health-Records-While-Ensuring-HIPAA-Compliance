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

---

## New interoperability + clinical workflow endpoints

### Patient Store additions (`http://patient-store:8002`)

- `GET /v1/patients/{patient_id}/fhir-bundle`
  - Exports a FHIR R4 Bundle (collection) containing Patient + mapped Observation/Condition/MedicationStatement/Procedure + DocumentReference resources.

- `POST /v1/fhir-bundle/import`
  - Imports a minimal FHIR Bundle and creates one patient + mapped events.

- `GET /v1/patients/{patient_id}/care-gaps`
  - Returns rule-based care gaps with severity and rationale.

- `GET /v1/patients/{patient_id}/summary`
  - Returns a snapshot payload with masked patient profile, recent events/documents, and care gaps.

- `POST /v1/cohorts`
  - Create saved cohort definition (`name`, `filters`).

- `GET /v1/cohorts`
  - List saved cohorts.

- `GET /v1/cohorts/{cohort_id}/patients`
  - Expands cohort to matched patients based on filter rules.

### Inference additions (`http://inference:8003`)

- `POST /v1/predict`
  - Now also returns:
    - `predictions[].explanation`: concise top contributing clinical factors (surrogate explainability)
    - `timeline` (when `patient_id` is used): longitudinal risk points + trend label

- `GET /v1/inference/metrics`
  - Telemetry summary (request counts + avg top probability over last 24h)

- `GET /v1/inference/drift`
  - Drift monitor based on 7-day vs prior 23-day score distribution deltas.

- `GET /v1/patients/{patient_id}/risk-timeline`
  - Patient-specific longitudinal risk timeline.

### RBAC + PHI masking

- Patient Store enforces role checks for JWT-backed user calls.
- PHI masking is applied on patient outputs unless caller has one of `PHI_UNMASK_ROLES` (default: `admin,clinician`).
