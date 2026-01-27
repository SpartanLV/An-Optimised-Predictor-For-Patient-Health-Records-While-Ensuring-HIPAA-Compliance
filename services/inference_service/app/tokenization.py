import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Keep exactly the same token patterns as your notebook
_ALLOWED_CHARS = re.compile(r"[^a-z0-9:_\-\.\+]+")

def clean_token_value(x: Any, max_len: int = 80) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = _ALLOWED_CHARS.sub("", s)
    s = s.strip("_")
    if not s:
        return ""
    return s[:max_len]

def norm_col_name(col: str) -> str:
    s = str(col).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def make_token(prefix: str, col: str, raw_value: Any) -> str:
    if raw_value is None:
        return ""
    v = clean_token_value(raw_value, max_len=80)
    if not v:
        return ""
    return f"{prefix}_{norm_col_name(col)}:{v}"

def recency_token(event_dt: datetime, index_dt: datetime) -> str:
    # Same bins as notebook
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)
    if index_dt.tzinfo is None:
        index_dt = index_dt.replace(tzinfo=timezone.utc)
    delta_days = (index_dt.date() - event_dt.date()).days
    if delta_days < 0:
        return ""
    if delta_days <= 7:
        return "time:0_7d"
    if delta_days <= 30:
        return "time:8_30d"
    if delta_days <= 90:
        return "time:31_90d"
    if delta_days <= 180:
        return "time:91_180d"
    if delta_days <= 365:
        return "time:181_365d"
    return "time:366_730d"

def obs_value_bin_token(code_or_desc: str, val: Any, bins: List[float]) -> str:
    try:
        x = float(val)
    except Exception:
        return ""
    if x != x:  # NaN
        return ""
    b = None
    for i in range(len(bins) - 1):
        if bins[i] <= x < bins[i + 1]:
            b = i
            break
    if b is None:
        return ""
    key = clean_token_value(code_or_desc, max_len=30) or "unknown"
    return f"obs_valbin:{key}_b{b}"

@dataclass
class FeatureSpec:
    prefix: str
    token_cols: List[str]

def build_tokens_from_events(
    events: List[Dict[str, Any]],
    as_of: datetime,
    cfg: Dict[str, Any],
) -> List[str]:
    """
    Replicates your notebook's patient tokenization:
      - recency token per event
      - src token
      - per-table tokens from configured columns
      - evt_sep token between events
      - truncation to max_tokens_per_patient
    """
    # Build prefix -> FeatureSpec mapping
    prefix_specs: Dict[str, FeatureSpec] = {}
    for _, spec in cfg.get("feature_tables", {}).items():
        prefix = spec["prefix"]
        prefix_specs[prefix] = FeatureSpec(prefix=prefix, token_cols=list(spec.get("token_cols", [])))

    lookback_days = int(cfg.get("lookback_days", 730))
    max_events = int(cfg.get("max_events_per_patient", 250))
    max_tokens_event = int(cfg.get("max_tokens_per_event", 10))
    max_tokens_patient = int(cfg.get("max_tokens_per_patient", 512))
    evt_sep = cfg.get("event_sep_token", "evt_sep")
    obs_bins = cfg.get("obs_value_bins", [])

    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    # Filter events within lookback window and <= as_of
    start_dt = as_of - timedelta(days=lookback_days)
    filtered = []
    for e in events:
        try:
            dt = e.get("event_date")
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if dt > as_of:
            continue
        if dt < start_dt:
            continue
        e2 = dict(e)
        e2["event_date"] = dt
        filtered.append(e2)

    # Keep most recent max_events
    filtered.sort(key=lambda x: x["event_date"])
    if len(filtered) > max_events:
        filtered = filtered[-max_events:]

    tokens: List[str] = []
    for e in filtered:
        prefix = (e.get("source") or "").strip()
        spec = prefix_specs.get(prefix)
        if not spec:
            # Unknown event source; skip
            continue

        local: List[str] = []
        tbin = recency_token(e["event_date"], as_of)
        if tbin:
            local.append(tbin)
        local.append(f"src:{prefix}")

        # map required cols to event dict keys
        col_map = {
            "CODE": "code",
            "DESCRIPTION": "description",
            "REASONCODE": "reason_code",
            "REASONDESCRIPTION": "reason_description",
            "TYPE": "type",
            "VALUE": "value",
            "UNITS": "units",
        }

        for col in spec.token_cols:
            raw_val = e.get(col_map.get(col, col.lower()))
            tok = ""
            if prefix == "obs" and col == "VALUE":
                key = e.get("code") or e.get("description") or ""
                tok = obs_value_bin_token(key, raw_val, obs_bins)
            else:
                tok = make_token(prefix, col, raw_val)
            if tok:
                local.append(tok)
            if len(local) >= max_tokens_event:
                break

        tokens.extend(local)
        tokens.append(evt_sep)

    # Truncate to last max_tokens_patient tokens (most recent info kept)
    if len(tokens) > max_tokens_patient:
        tokens = tokens[-max_tokens_patient:]

    return tokens
