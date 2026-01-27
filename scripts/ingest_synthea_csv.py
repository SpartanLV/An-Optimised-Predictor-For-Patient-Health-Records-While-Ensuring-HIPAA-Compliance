#!/usr/bin/env python3
"""Synthea CSV → Patient Store ingest (demo).

Reads Synthea's CSV output folder and inserts:
  - patients
  - events (encounters, observations, medications, procedures, careplans, immunizations, allergies)

This is intentionally lightweight (stdlib-only) and meant for coursework/demo.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            cleaned: Dict[str, str] = {}
            for k, v in row.items():
                if k is None:
                    # Extra columns beyond header -> DictReader stores them under key None as a list
                    # We ignore them (or you could join them if you want to debug)
                    continue
                if isinstance(v, list):
                    v = " ".join(v)
                cleaned[k] = (v or "").strip()
            yield cleaned



def _parse_dt(value: str) -> str:
    """Return an ISO-8601 timestamp with timezone.

    Synthea commonly emits:
      - 2016-01-01
      - 2016-01-01T12:34:56Z
      - 2016-01-01T12:34:56
    """
    v = (value or "").strip()
    if not v:
        # fall back to now UTC
        return dt.datetime.now(dt.timezone.utc).isoformat()

    # Date only
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        d = dt.date.fromisoformat(v)
        return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).isoformat()

    # With Z
    if v.endswith("Z"):
        try:
            # Python can't parse 'Z' in fromisoformat until 3.11? normalize
            v2 = v[:-1] + "+00:00"
            return dt.datetime.fromisoformat(v2).isoformat()
        except Exception:
            pass

    # With timezone or naive
    try:
        ts = dt.datetime.fromisoformat(v)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return ts.isoformat()
    except Exception:
        # last resort: now
        return dt.datetime.now(dt.timezone.utc).isoformat()


def _http_json(req: urllib.request.Request) -> Tuple[int, str]:
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return e.code, body


def _get_token(auth_url: str, username: str, password: str) -> str:
    form = urllib.parse.urlencode({"username": username, "password": password}).encode("utf-8")
    req = urllib.request.Request(
        f"{auth_url.rstrip('/')}/v1/auth/token",
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    code, body = _http_json(req)
    if code != 200:
        raise RuntimeError(f"Auth token failed: HTTP {code}: {body[:500]}")
    return json.loads(body)["access_token"]


def _create_patient(base_url: str, token: str, patient: Dict[str, Any]) -> Optional[str]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/patients",
        data=json.dumps(patient).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    code, body = _http_json(req)
    if code == 200:
        return json.loads(body)["id"]
    # duplicate MRN or other errors are okay for demo
    sys.stderr.write(f"[patient] skip HTTP {code}: {body[:200]}\n")
    return None


def _bulk_events(base_url: str, token: str, patient_id: str, events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/patients/{patient_id}/events:bulk",
        data=json.dumps({"events": events}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    code, body = _http_json(req)
    if code != 200:
        raise RuntimeError(f"Bulk events failed for {patient_id}: HTTP {code}: {body[:500]}")


def _event(source: str, when: str, code: str = "", desc: str = "", value: str = "", units: str = "", typ: str = "") -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "source": source,
        "event_date": _parse_dt(when),
        "code": code or None,
        "description": desc or None,
        "value": value or None,
        "units": units or None,
        "type": typ or None,
    }
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="Path to Synthea output/csv directory")
    ap.add_argument("--base-url", required=True, help="Patient-store base URL (e.g., http://patient-store:8002 or https://localhost:8443/api/store)")
    ap.add_argument("--auth-url", required=True, help="Auth base URL (e.g., http://auth:8004 or https://localhost:8443/auth)")
    ap.add_argument("--username", default=os.getenv("ADMIN_USERNAME", "admin"))
    ap.add_argument("--password", default=os.getenv("ADMIN_PASSWORD", "admin123"))
    ap.add_argument("--limit-patients", type=int, default=100)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        raise SystemExit(f"CSV dir not found: {csv_dir}")

    token = _get_token(args.auth_url, args.username, args.password)

    # 1) create patients
    patients_path = csv_dir / "patients.csv"
    if not patients_path.exists():
        raise SystemExit(f"Missing {patients_path}")

    syn_to_uuid: Dict[str, str] = {}
    created = 0
    for row in _read_csv(patients_path):
        if created >= args.limit_patients:
            break
        syn_id = row.get("Id") or row.get("ID") or row.get("id")
        if not syn_id:
            continue
        patient = {
            "mrn": syn_id,
            "first_name": row.get("FIRST") or row.get("First") or row.get("FIRSTNAME") or row.get("FIRST_NAME"),
            "last_name": row.get("LAST") or row.get("Last") or row.get("LASTNAME") or row.get("LAST_NAME"),
            "dob": row.get("BIRTHDATE") or row.get("DOB") or row.get("BIRTH_DATE"),
            "gender": row.get("GENDER") or row.get("SEX"),
        }
        pid = _create_patient(args.base_url, token, patient)
        if pid:
            syn_to_uuid[syn_id] = pid
            created += 1

    if not syn_to_uuid:
        sys.stderr.write("No patients created (check credentials / base URLs).\n")
        return 2

    # 2) collect events (grouped by patient)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_events(path: Path, source: str, date_key: str) -> None:
        if not path.exists():
            return
        for row in _read_csv(path):
            syn_id = row.get("PATIENT") or row.get("Patient") or row.get("patient")
            if not syn_id or syn_id not in syn_to_uuid:
                continue
            when = row.get(date_key) or ""
            code = row.get("CODE") or row.get("Code") or ""
            desc = row.get("DESCRIPTION") or row.get("Description") or ""
            value = row.get("VALUE") or row.get("Value") or ""
            units = row.get("UNITS") or row.get("Units") or ""
            typ = row.get("TYPE") or row.get("Type") or ""
            buckets[syn_to_uuid[syn_id]].append(_event(source, when, code=code, desc=desc, value=value, units=units, typ=typ))

    # Encounters typically have START
    add_events(csv_dir / "encounters.csv", "enc", "START")
    # Observations typically have DATE
    add_events(csv_dir / "observations.csv", "obs", "DATE")
    # Medications typically have START
    add_events(csv_dir / "medications.csv", "med", "START")
    add_events(csv_dir / "procedures.csv", "proc", "DATE")
    add_events(csv_dir / "careplans.csv", "cp", "START")
    add_events(csv_dir / "immunizations.csv", "imm", "DATE")
    add_events(csv_dir / "allergies.csv", "alg", "START")

    # 3) push events
    total_events = 0
    for pid, evs in buckets.items():
        _bulk_events(args.base_url, token, pid, evs)
        total_events += len(evs)

    print(json.dumps({"patients_created": len(syn_to_uuid), "events_inserted": total_events}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
