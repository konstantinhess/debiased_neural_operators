from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from typing import Iterator

try:
    import mlflow
except ImportError as exc:  # pragma: no cover - exercised by runtime setup
    mlflow = None
    _MLFLOW_IMPORT_ERROR = exc
else:
    _MLFLOW_IMPORT_ERROR = None


_MODEL_ARTIFACT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx"}


def mlflow_import_error() -> Exception | None:
    return _MLFLOW_IMPORT_ERROR


class MlflowTracker:
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        enabled: bool = True,
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = enabled and mlflow is not None and os.getenv("ONO_ENABLE_MLFLOW", "1") != "0"
        self.explicit_tracking_uri = tracking_uri is not None or "MLFLOW_TRACKING_URI" in os.environ
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5555")
        self._active = False

    def start(self, tags: dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_name, tags=tags or {})
            self._active = True
        except Exception as exc:  # pragma: no cover - depends on local server state
            if not self.explicit_tracking_uri:
                warnings.warn(
                    "MLflow logging was skipped because the default local tracking server "
                    "at http://127.0.0.1:5555 was not reachable.",
                    stacklevel=2,
                )
                self.enabled = False
                return
            raise RuntimeError(
                "MLflow logging is enabled but the tracking server could not be reached. "
                "Start the local MLflow server from the repo root and set MLFLOW_TRACKING_URI "
                "to http://127.0.0.1:5555 if needed."
            ) from exc

    def end(self) -> None:
        if self.enabled and self._active:
            assert mlflow is not None
            mlflow.end_run()
            self._active = False

    @contextmanager
    def nested_run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        assert mlflow is not None
        with mlflow.start_run(run_name=run_name, nested=True, tags=tags or {}):
            yield

    def log_params_from_dict(self, prefix: str, payload: dict) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        flat_params = _flatten_params(prefix, payload)
        if flat_params:
            mlflow.log_params(flat_params)

    def log_metrics_from_dict(self, payload: dict, prefix: str = "") -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        metrics = _flatten_metrics(prefix, payload)
        if metrics:
            mlflow.log_metrics(metrics)

    def log_json_artifact(self, payload: dict, artifact_file: str) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        temp_root = Path(".mlflow_tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        artifact_path = temp_root / artifact_file
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        artifact_parent = Path(artifact_file).parent
        if str(artifact_parent) in {"", "."}:
            mlflow.log_artifact(str(artifact_path))
        else:
            mlflow.log_artifact(str(artifact_path), artifact_path=str(artifact_parent))

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        path = Path(path)
        if _is_model_artifact(path):
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        assert mlflow is not None
        path = Path(path)
        if path.is_file():
            self.log_artifact(path, artifact_path=artifact_path)
            return
        if not path.exists():
            return
        temp_parent = Path(".mlflow_tmp")
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="filtered_artifacts_", dir=temp_parent))
        try:
            filtered_root = temp_root / path.name
            copied_any = False
            for source in path.rglob("*"):
                if not source.is_file() or _is_model_artifact(source):
                    continue
                destination = filtered_root / source.relative_to(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied_any = True
            if copied_any:
                mlflow.log_artifacts(str(filtered_root), artifact_path=artifact_path)
        finally:
            _safe_rmtree(temp_root)


def _is_model_artifact(path: Path) -> bool:
    return path.suffix.lower() in _MODEL_ARTIFACT_SUFFIXES


def _safe_rmtree(path: Path, retries: int = 5, delay_seconds: float = 0.2) -> None:
    if not path.exists():
        return
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    if last_error is not None:
        warnings.warn(
            f"Could not remove temporary MLflow artifact directory: {path}",
            stacklevel=2,
        )


def _flatten_params(prefix: str, payload: dict) -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_params(name, value))
        elif isinstance(value, list):
            flat[name] = json.dumps(value)
        elif value is None:
            flat[name] = "null"
        else:
            flat[name] = str(value)
    return flat


def _flatten_metrics(prefix: str, payload: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(name, value))
        elif isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat
