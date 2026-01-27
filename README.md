# CSE400 HIPAA-Oriented Microservices Platform (OCR/NER → Store → Hybrid HMM + CNN‑BiLSTM + Fusion)

This repo is a **working, locally runnable** microservices platform that matches your 3-stage pipeline:

1. **Stage 1 – Preprocess Service**: medical document upload → PDF text extraction + OCR → best‑effort NER + event hints  
2. **Stage 2 – Patient Store Service**: stores patients + documents + extracted entities + structured events (Postgres)  
3. **Stage 3 – Inference Service**: loads your exported **CNN‑BiLSTM multilabel model + HMM feature extractor + fusion MLP** and returns predictions + recommendations

It also includes a minimal **Auth service** (JWT) and an HTTPS **Gateway** (TLS termination) so the web platform is closer to HIPAA-aligned technical safeguards.

> ⚠️ HIPAA reality check: HIPAA compliance is **not just code**. You must also do **risk analysis**, policies, training, access governance, BAAs (if applicable), and operational controls.  
> Engineering references: 45 CFR § 164.312 (technical safeguards) and 45 CFR § 164.308 (risk analysis). See `docs/HIPAA_COMPLIANCE.md`.

---

## Services

- `preprocess_service` (Stage 1) – FastAPI, PyMuPDF + Tesseract OCR
- `patient_store_service` (Stage 2) – FastAPI + SQLAlchemy + Postgres (+ optional at-rest file encryption)
- `inference_service` (Stage 3) – FastAPI + TensorFlow + hmmlearn + fusion model
- `auth_service` – issues JWTs (demo; replace with real OIDC provider later)
- `web_portal` – simple UI orchestrator (server-side)
- `gateway` – Caddy reverse proxy with **HTTPS** (`tls internal`)

---
## Quickerstart if you do not want to manually do things

### 1) Prereqs
- Docker Desktop

### 2) Double click on "RUN_ME.bat"


## Quickstart (Secure Mode – recommended)

### 1) Prereqs
- Docker Desktop

### 2) Configure secrets
Copy `.env.example` → `.env`, then set:
- `API_KEY` (internal service-to-service)
- `JWT_SECRET` (strong random)
- `ADMIN_PASSWORD` (change default)
- (optional) `FILE_ENCRYPTION_KEY` to encrypt stored documents

### 3) Start
```bash
docker compose up --build
```

### 4) Open the web platform (HTTPS)
- https://localhost:8443

Login with:
- Username: `ADMIN_USERNAME` (default `admin`)
- Password: `ADMIN_PASSWORD`

---

## Dev Mode (ports exposed, HTTP)

If you want to view per-service Swagger docs directly:
```bash
docker compose -f docker-compose.dev.yml up --build
```

Then:
- Web Portal: http://localhost:8000
- Auth: http://localhost:8004/docs
- Preprocess: http://localhost:8001/docs
- Patient Store: http://localhost:8002/docs
- Inference: http://localhost:8003/docs

---

## Model bundle

Your Stage‑3 exported artifacts live in `./model_bundle/`, including:
- `deep_cnn_bilstm_multilabel.keras`
- `fusion_mlp.keras`
- `hmm_feature_extractor.joblib`
- `hmm_feature_scaler.joblib`
- `tokenizer.joblib`
- `multilabel_binarizer.joblib`
- `metadata.json`, `cfg.json`, `label_code_to_description.json`, `recommendations.json`

To swap in a newer export, overwrite files in `model_bundle/`.

---

## Notes on PHI / “minimum necessary”

- Stage 2 does **not** store raw OCR text (only documents + extracted entities + events).
- Stage 1 can be configured to **not return OCR text** (`RETURN_OCR_TEXT=false`) to reduce PHI movement.
- Avoid enabling `return_debug=true` in inference outside debugging; it can include token snippets.

---

## Next steps (recommended roadmap)

1) Replace demo NER with clinical NLP (medSpaCy / scispaCy / HF biomedical NER) running **in your secure environment**  
2) Replace demo auth with OIDC (Keycloak/Okta/Azure AD) + RBAC  
3) Add FHIR mapping (Patient, Encounter, Observation, MedicationRequest, Condition)  
4) Add queue-based ingestion (for scale and reliability) and PHI-safe observability


## SyntheaMass CSV ingestion (optional)

If you generated Synthea output in CSV format, you can load a small subset into Stage‑2 using:

```bash
python scripts/ingest_synthea_csv.py --csv-dir /path/to/synthea/output/csv \
  --base-url http://localhost:8002 --auth-url http://localhost:8004 \
  --username admin --password <ADMIN_PASSWORD> --limit-patients 100
```

If you’re running in secure mode behind the gateway, point both `base-url` and `auth-url` to the gateway and include the correct path prefixes:

```bash
python scripts/ingest_synthea_csv.py --csv-dir /path/to/synthea/output/csv \
  --base-url https://localhost:8443/api/store --auth-url https://localhost:8443/auth \
  --username admin --password <ADMIN_PASSWORD> --insecure-tls
```


## Windows note: trusting the Caddy dev certificate (optional)

The gateway uses `tls internal` (a self-signed local CA). Browsers and PowerShell may warn until you trust it.

1) Copy the CA cert out of the gateway container:

```powershell
$gid = docker compose ps -q gateway
docker cp ${gid}:/data/caddy/pki/authorities/local/root.crt .\caddy_root.crt
```

2) Import it into the Windows Trusted Root store (run PowerShell **as Administrator**):

```powershell
Import-Certificate -FilePath .\caddy_root.crt -CertStoreLocation Cert:\LocalMachine\Root
```

If you **don’t have admin rights**, you can import it for just your user:

```powershell
Import-Certificate -FilePath .\caddy_root.crt -CertStoreLocation Cert:\CurrentUser\Root
```

Or use the helper script (no admin required by default):

```powershell
# installs to Cert:\CurrentUser\Root
.\scripts\install_caddy_cert_currentuser.ps1 -CertPath .\caddy_root.crt

# if you *are* running an elevated (Admin) shell and want LocalMachine:
.\scripts\install_caddy_cert_currentuser.ps1 -CertPath .\caddy_root.crt -LocalMachine
```


For production you should use a real certificate (e.g., via Let’s Encrypt or your org’s PKI) and enforce TLS 1.2+.
