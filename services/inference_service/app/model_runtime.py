import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import httpx

import zipfile
import tempfile

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .tokenization import build_tokens_from_events
from .schemas import RecommendationsOut, RecommendationItem

# Keep tensorflow logs quieter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_keras_model_portable(model_path: Path):
    """Load a .keras model robustly across Windows->Linux.

    Some Windows-saved .keras files may contain backslashes in internal HDF5 paths
    (e.g., 'layers\\dense') which makes Linux containers fail to locate weights.
    If a normal load fails, we rewrite the weights file into a POSIX-safe layout
    in a temp directory and retry.

    This keeps the rest of your pipeline unchanged.
    """
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        # Only attempt auto-repair for the new Keras zip format
        if str(model_path).lower().endswith(".keras"):
            return _load_keras_with_portability_patch(model_path)
        raise


def _convert_h5_name(name: str, embed_alias: Optional[str] = None) -> str:
    # HDF5 uses '/' for hierarchy, but some Windows-saved artifacts incorrectly
    # use '\\' inside names. Convert both into a POSIX-like path.
    parts: List[str] = []
    for seg in name.split("/"):
        if not seg:
            continue
        for p in seg.split("\\"):
            if p:
                parts.append(p)

    # Optional: if the model config expects an embedding layer named 'embed'
    # but weights were stored under 'embedding', alias it.
    if embed_alias:
        for i in range(len(parts) - 1):
            if parts[i] == "layers" and parts[i + 1] == "embedding":
                parts[i + 1] = embed_alias
    return "/".join(parts)


def _patch_weights_h5_posix(src_h5: Path, dst_h5: Path, embed_alias: Optional[str] = None) -> None:
    import h5py  # tensorflow pulls this in already
    import numpy as _np

    def copy_attrs(src_obj, dst_obj):
        for k, v in src_obj.attrs.items():
            try:
                dst_obj.attrs[k] = v
            except Exception:
                pass

    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        copy_attrs(src, dst)

        def visitor(name, obj):
            new_name = _convert_h5_name(name, embed_alias=embed_alias)

            if isinstance(obj, h5py.Group):
                g = dst.require_group(new_name) if new_name else dst
                copy_attrs(obj, g)
                return

            if isinstance(obj, h5py.Dataset):
                # ensure parent groups exist
                parent = new_name.rsplit("/", 1)[0] if "/" in new_name else ""
                if parent:
                    dst.require_group(parent)

                if new_name in dst:
                    return

                data = obj[()]
                # create dataset
                dset = dst.create_dataset(new_name, data=data)
                copy_attrs(obj, dset)

        src.visititems(visitor)


def _load_keras_with_portability_patch(model_path: Path):
    # Patch into a temp dir, load, then return the in-memory model.
    with tempfile.TemporaryDirectory(prefix="keras_portable_") as td:
        td = Path(td)
        extract_dir = td / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(model_path, "r") as z:
            z.extractall(extract_dir)

        cfg = json.loads((extract_dir / "config.json").read_text(encoding="utf-8"))

        def _replace_exact_string(obj: Any, old: str, new: str) -> Any:
            if isinstance(obj, dict):
                return {k: _replace_exact_string(v, old, new) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_replace_exact_string(v, old, new) for v in obj]
            if obj == old:
                return new
            return obj

        def _layer_names_from_cfg(cfg_obj: Dict[str, Any]) -> List[str]:
            try:
                return [l.get("config", {}).get("name") for l in cfg_obj.get("config", {}).get("layers", [])]
            except Exception:
                return []

        layer_names = _layer_names_from_cfg(cfg)

        weights_path = extract_dir / "model.weights.h5"

        # Detect the Windows-backslash issue quickly
        needs_patch = False
        embed_name_mismatch = False
        try:
            import h5py
            with h5py.File(weights_path, "r") as h5:
                # Windows-saved artifacts often end up with backslashes in group names.
                for k in h5.keys():
                    if "\\" in k:
                        needs_patch = True
                        break

                # Some Windows-saved .keras files end up with config expecting an Embedding
                # layer named 'embed', but the weights are stored under 'embedding'.
                cfg_has_embed = ("embed" in layer_names) and ("embedding" not in layer_names)
                if cfg_has_embed:
                    has_layers_embedding = (
                        any(k.startswith("layers\\embedding") or k.startswith("layers/embedding") for k in h5.keys())
                        or ("layers" in h5 and "embedding" in h5["layers"])
                    )
                    has_layers_embed = (
                        any(k.startswith("layers\\embed") or k.startswith("layers/embed") for k in h5.keys())
                        or ("layers" in h5 and "embed" in h5["layers"])
                    )
                    embed_name_mismatch = has_layers_embedding and (not has_layers_embed)
        except Exception:
            needs_patch = True

        # If we detected a mismatch (config expects 'embed' but weights live under 'embedding'),
        # prefer renaming the WEIGHTS paths to match the config. This avoids relying on config edits.
        embed_alias = "embed" if embed_name_mismatch else None


        if needs_patch or embed_alias:
            patched_h5 = extract_dir / "model.weights.patched.h5"
            _patch_weights_h5_posix(weights_path, patched_h5, embed_alias=embed_alias)
            weights_path.unlink(missing_ok=True)
            patched_h5.rename(weights_path)

        patched_model = td / ("patched_" + model_path.name)
        with zipfile.ZipFile(patched_model, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for fname in ["metadata.json", "config.json", "model.weights.h5"]:
                z.write(extract_dir / fname, arcname=fname)

        return tf.keras.models.load_model(patched_model, compile=False)



class ModelBundle:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir

        # Load json config/metadata
        self.cfg = json.loads((bundle_dir / "cfg.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        self.code_to_desc = json.loads((bundle_dir / "label_code_to_description.json").read_text(encoding="utf-8"))
        self.recommendations = json.loads((bundle_dir / "recommendations.json").read_text(encoding="utf-8"))

        # Threshold to use for inference
        self.threshold = float(self.metadata.get("thresholds", {}).get("use_for_inference", {}).get("thr", 0.3))

        # Labels in the same order as model outputs
        self.label_names: List[str] = list(self.metadata.get("labels", {}).get("label_names", []))

        # Load ML artifacts
        self.deep_model = load_keras_model_portable(bundle_dir / self.metadata["models"]["deep_model_path"])
        self.fusion_model = load_keras_model_portable(bundle_dir / self.metadata["models"]["fusion_model_path"])

        self.tokenizer = joblib.load(bundle_dir / "tokenizer.joblib")
        self.mlb = joblib.load(bundle_dir / "multilabel_binarizer.joblib")
        self.hmm = joblib.load(bundle_dir / "hmm_feature_extractor.joblib")
        self.hmm_scaler = joblib.load(bundle_dir / "hmm_feature_scaler.joblib")

        # Derived sizes
        self.max_seq_len = int(self.cfg.get("max_seq_len", 512))

    def _text_to_padded_seq(self, text: str) -> np.ndarray:
        seq = self.tokenizer.texts_to_sequences([text])
        X = pad_sequences(seq, maxlen=self.max_seq_len, padding="post", truncating="post")
        return X.astype(np.int32)

    def _hmm_extract_features(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Matches your notebook's hmm_extract_features:
          - avg_loglik
          - (optional length)
          - posterior state occupancy mean
        """
        cfg = self.cfg
        n_states = int(getattr(self.hmm, "n_components", 0)) or int(cfg.get("hmm_states", 8))
        use_len = bool(cfg.get("hmm_feat_use_length", True))
        feat_max_len = cfg.get("hmm_feat_max_len", 256)

        extra = 1 + (1 if use_len else 0)
        feat_dim = extra + n_states
        feats = np.zeros((len(X_seq), feat_dim), dtype=np.float32)

        for i, row in enumerate(X_seq):
            s = row[row > 0].astype(int)
            if len(s) == 0:
                feats[i, 0] = -50.0
                if use_len:
                    feats[i, 1] = 0.0
                continue

            if feat_max_len is not None:
                try:
                    s = s[-int(feat_max_len):]
                except Exception:
                    pass

            X = s.reshape(-1, 1)
            L = max(len(s), 1)

            try:
                ll = float(self.hmm.score(X))
                avg_ll = ll / L
            except Exception:
                avg_ll = -50.0

            feats[i, 0] = avg_ll
            offset = 1
            if use_len:
                feats[i, 1] = float(L)
                offset = 2

            try:
                gamma = self.hmm.predict_proba(X)  # (L, n_states)
                occ = gamma.mean(axis=0)
            except Exception:
                st = self.hmm.predict(X)
                occ = np.bincount(st, minlength=n_states) / max(len(st), 1)

            feats[i, offset:offset + n_states] = occ.astype(np.float32)

        # Safety
        feats = np.nan_to_num(feats, nan=-50.0, posinf=50.0, neginf=-50.0)
        return feats

    def recommend_for_code(self, code: str, top_k: int = 5) -> RecommendationsOut:
        r = self.recommendations.get(code, {}) if isinstance(self.recommendations, dict) else {}
        meds = [RecommendationItem(**x) for x in (r.get("medications") or [])[:top_k]]
        procs = [RecommendationItem(**x) for x in (r.get("procedures") or [])[:top_k]]
        cps = [RecommendationItem(**x) for x in (r.get("careplans") or [])[:top_k]]
        return RecommendationsOut(medications=meds, procedures=procs, careplans=cps)

    def predict_from_events(self, events: List[Dict[str, Any]], as_of: Optional[datetime] = None, top_n: int = 10, return_debug: bool = False) -> Dict[str, Any]:
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        tokens = build_tokens_from_events(events=events, as_of=as_of, cfg=self.cfg)
        text = " ".join(tokens)

        X_seq = self._text_to_padded_seq(text)

        # Deep probs
        p_deep = self.deep_model.predict(X_seq, verbose=0)
        p_deep = np.asarray(p_deep, dtype=np.float32)

        # HMM features (scaled)
        hmm_feats = self._hmm_extract_features(X_seq)
        hmm_feats_s = self.hmm_scaler.transform(hmm_feats).astype(np.float32)

        # Fused probs
        p_fused = self.fusion_model.predict([p_deep, hmm_feats_s], verbose=0)
        p_fused = np.asarray(p_fused, dtype=np.float32)

        probs = p_fused[0]
        thr = self.threshold

        # Build predictions
        items = []
        for i, code in enumerate(self.label_names):
            pr = float(probs[i])
            if pr >= thr:
                items.append((code, pr))
        items.sort(key=lambda x: x[1], reverse=True)

        # If nothing crosses threshold, return top 3 as low-confidence signal
        if not items:
            top_idx = np.argsort(-probs)[:min(3, len(probs))]
            items = [(self.label_names[int(i)], float(probs[int(i)])) for i in top_idx]

        # Trim to top_n
        items = items[:max(1, int(top_n))]

        out_preds = []
        for code, pr in items:
            out_preds.append({
                "code": code,
                "description": self.code_to_desc.get(code),
                "probability": pr,
                "recommendations": self.recommend_for_code(code).model_dump(),
            })

        debug = None
        if return_debug:
            debug = {
                "as_of": as_of.isoformat(),
                "num_events": len(events),
                "num_tokens": len(tokens),
                "sample_tokens_tail": tokens[-30:],
                "hmm_feats_raw": hmm_feats[0].tolist(),
                "threshold": thr,
            }

        return {
            "threshold": thr,
            "predictions": out_preds,
            "debug": debug,
        }


async def fetch_events_from_store(patient_store_url: str, api_key: str, patient_id: str, request_id: str | None = None) -> List[Dict[str, Any]]:
    url = f"{patient_store_url.rstrip('/')}/v1/patients/{patient_id}/events"
    headers = {"X-API-Key": api_key, "X-User-Id": "inference-service"}
    if request_id:
        headers["X-Request-ID"] = request_id
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=headers, params={"limit": 5000})
        r.raise_for_status()
        return r.json()
