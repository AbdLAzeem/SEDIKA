"""Versioned artifact persistence for stateful preprocessing objects.

A bare `joblib.dump(scaler, ...)` discards everything that makes the artifact
interpretable later: which feature columns were fit, in what order, against
which library version. If the upstream feature list shifts by even one column,
inference silently mis-projects every sample.

`save_artifact` wraps the object in a small manifest. `load_artifact`
re-validates the manifest on read — by default it raises on mismatch so the
failure is loud rather than silent.
"""
from __future__ import annotations

import os
import sys
import hashlib
import datetime as _dt
from typing import Iterable, Optional

import joblib
import sklearn

MANIFEST_VERSION = 1


def _hash_features(feature_names: Iterable[str]) -> str:
    h = hashlib.sha256()
    for name in feature_names:
        h.update(str(name).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def save_artifact(obj, path: str, *, feature_names: Iterable[str],
                  kind: str, extra: Optional[dict] = None) -> dict:
    """Persist `obj` plus a manifest at `path` (single .joblib file).

    Args:
        obj: the fitted sklearn-like object (scaler, encoder, etc.)
        path: target file path (.joblib)
        feature_names: column names in the order the artifact was fit on
        kind: free-form tag, e.g. "scaler", "label_encoder", "rf_selector"
        extra: optional dict of additional metadata
    """
    feature_names = list(feature_names)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "kind": kind,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "feature_hash": _hash_features(feature_names),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "saved_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "extra": dict(extra or {}),
    }
    payload = {"manifest": manifest, "object": obj}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(payload, path)
    return manifest


def load_artifact(path: str, *, expected_features: Optional[Iterable[str]] = None,
                  strict: bool = True):
    """Load an artifact and validate its manifest.

    Returns the unwrapped object. Raises ArtifactMismatchError if
    `expected_features` is supplied and does not match the saved manifest.

    `strict=False` downgrades the mismatch to a warning (returned in `.warnings`
    on the manifest dict, accessible via `load_artifact_with_manifest`).
    """
    obj, _ = load_artifact_with_manifest(path, expected_features=expected_features, strict=strict)
    return obj


def load_artifact_with_manifest(path: str, *,
                                expected_features: Optional[Iterable[str]] = None,
                                strict: bool = True):
    payload = joblib.load(path)

    # Backward compatibility: a bare object saved with raw joblib.dump.
    if not isinstance(payload, dict) or "manifest" not in payload or "object" not in payload:
        if strict and expected_features is not None:
            raise ArtifactMismatchError(
                f"{path}: artifact has no manifest; cannot validate feature schema. "
                f"Re-save with artifacts.save_artifact()."
            )
        return payload, {"manifest_version": 0, "warnings": ["no_manifest"]}

    manifest = payload["manifest"]
    obj = payload["object"]

    if expected_features is not None:
        expected = list(expected_features)
        saved = manifest.get("feature_names", [])
        if expected != saved:
            msg = (
                f"{path}: feature schema mismatch.\n"
                f"  expected ({len(expected)}): {expected[:6]}{'...' if len(expected) > 6 else ''}\n"
                f"  saved    ({len(saved)}):    {saved[:6]}{'...' if len(saved) > 6 else ''}"
            )
            if strict:
                raise ArtifactMismatchError(msg)
            manifest.setdefault("warnings", []).append("feature_mismatch")
            print(f"WARNING: {msg}")

    return obj, manifest


class ArtifactMismatchError(RuntimeError):
    """Raised when a loaded artifact's manifest disagrees with caller expectations."""
