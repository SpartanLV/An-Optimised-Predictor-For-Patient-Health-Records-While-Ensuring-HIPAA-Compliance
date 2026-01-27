# HIPAA Compliance Notes (Engineering-Facing)

This project is designed to be **HIPAA-oriented** (Security Rule + Privacy Rule aware), but **HIPAA compliance is not “a library you install.”**
It requires **technical + administrative + physical safeguards**, plus documentation and operational controls.

Use this document as a build checklist.

---

## What the repo implements (MVP)

### Technical safeguards (Security Rule – 45 CFR § 164.312)
- **Access control (baseline)**:
  - End-user authentication via **JWT** (demo Auth service).
  - Internal service-to-service authentication via `X-API-Key`.
- **Audit controls (baseline)**:
  - Patient Store writes audit log entries for reads/writes (who/what/when).
- **Integrity**:
  - Document SHA-256 hashing stored with each document record.
- **Transmission security**:
  - `gateway` terminates **HTTPS** (Caddy `tls internal`) in secure mode.
- **Optional at-rest file encryption**:
  - Patient Store can encrypt stored documents via `FILE_ENCRYPTION_KEY` (Fernet).

> NOTE: The Security Rule’s encryption specs are “addressable”; implement them when reasonable and appropriate based on your risk analysis.

### Privacy Rule awareness (Minimum Necessary)
- Stage 2 does not store raw OCR text (only files + extracted entities + events).
- Stage 1 can be configured to not return OCR text (`RETURN_OCR_TEXT=false`) to reduce PHI movement.

---

## What you still must do for real HIPAA deployments

### Administrative safeguards
- Formal **risk analysis** + risk management plan
- Policies: workforce training, incident response, access provisioning/deprovisioning
- BAAs with any cloud/hosting vendors (if you use them)

### Physical safeguards
- Device controls, facility access controls, secure backups/media disposal

### Technical safeguards (production-grade)
- Replace demo auth with enterprise OIDC/OAuth2 + RBAC/ABAC
- TLS everywhere (external + internal), ideally mTLS between services
- Centralized audit logging + alerting + immutable storage (WORM)
- Secrets management (Vault/KMS), key rotation
- Database encryption at rest (disk / managed DB) + encrypted backups
- Data retention & deletion policies
- Vulnerability scanning, patching, SBOM, dependency monitoring
