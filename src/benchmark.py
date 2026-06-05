"""Benchmark segmentation inference across runtime backends.

Outputs:
- results.csv
- results.md
- environment.json
"""

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path
from urllib import request

import numpy as np
import psutil
import torch

from export import export_to_onnx
from predict import (
    _build_model_from_checkpoint,
    _load_config_file,
    _load_volume,
    _predict_onnxruntime,
    _predict_pytorch,
)


def _discover_inputs(input_dir, max_cases):
    root = Path(input_dir)
    if root.is_file():
        return [root]

    image_candidates = sorted(root.rglob("image.npy"))
    if not image_candidates:
        all_candidates = [
            *sorted(root.rglob("*.npy")),
            *sorted(root.rglob("*.nii")),
            *sorted(root.rglob("*.nii.gz")),
        ]
        image_candidates = [p for p in all_candidates if "labels" not in p.name.lower()]

    if not image_candidates:
        raise ValueError(f"No inputs found in {input_dir}")

    if max_cases > 0:
        return image_candidates[:max_cases]
    return image_candidates


def _resolve_label_path(input_path, labels_dir):
    if labels_dir is None:
        return None

    input_path = Path(input_path)
    labels_root = Path(labels_dir)

    if input_path.name == "image.npy":
        candidate = labels_root / input_path.parent.name / "labels.npy"
        if candidate.exists():
            return candidate

        candidate = input_path.with_name("labels.npy")
        if candidate.exists():
            return candidate

    stem = input_path.stem
    if stem.endswith(".nii"):
        stem = stem[:-4]
    for suffix in [".npy", ".nii", ".nii.gz"]:
        candidate = labels_root / f"{stem}_label{suffix}"
        if candidate.exists():
            return candidate

    return None


def _foreground_dice(pred, ref):
    pred_fg = np.asarray(pred) > 0
    ref_fg = np.asarray(ref) > 0
    inter = float(np.logical_and(pred_fg, ref_fg).sum())
    denom = float(pred_fg.sum() + ref_fg.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + 1e-6) / (denom + 1e-6)


def _percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _to_model_size_mb(path):
    try:
        size_bytes = os.path.getsize(path)

        # ONNX export may store tensor data in a sidecar file: <model>.onnx.data
        sidecar = f"{path}.data"
        if str(path).endswith(".onnx") and os.path.isfile(sidecar):
            size_bytes += os.path.getsize(sidecar)

        return round(size_bytes / (1024.0 * 1024.0), 3)
    except OSError:
        return 0.0


def _collect_environment():
    env = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "onnxruntime_version": None,
        "cpu": platform.processor() or "unknown",
        "gpu": None,
        "ram_gb": round(psutil.virtual_memory().total / (1024.0**3), 2),
    }

    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)

    try:
        import onnxruntime as ort

        env["onnxruntime_version"] = ort.__version__
    except Exception:
        env["onnxruntime_version"] = "not-installed"

    return env


def _benchmark_backend(
    backend,
    inputs,
    labels_dir,
    warmup,
    device,
    model=None,
    config=None,
    session=None,
    input_name=None,
    api_url=None,
):
    process = psutil.Process(os.getpid())

    latency_ms = []
    dice_scores = []
    max_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
    output_shape_valid = True

    def run_once(input_path):
        volume, _ = _load_volume(str(input_path))

        if backend in {"pytorch_cpu", "pytorch_cuda"}:
            return _predict_pytorch(model, volume, config, device)
        if backend == "onnx_cpu":
            return _predict_onnxruntime(session, input_name, volume)
        if backend == "api_cpu":
            url = api_url.rstrip("/") + "/predict?include_summary=false"
            with open(input_path, "rb") as f:
                body = f.read()

            req = request.Request(url=url, method="POST", data=body)
            req.add_header("Content-Type", "application/octet-stream")
            # The simple API expects multipart upload; this backend remains optional.
            raise RuntimeError(
                "api_cpu backend expects multipart form upload. Use pytorch_cpu or onnx_cpu for local benchmark."
            )

        raise ValueError(f"Unknown backend: {backend}")

    for input_path in inputs:
        for _ in range(max(warmup, 0)):
            _ = run_once(input_path)

        if backend == "pytorch_cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = run_once(input_path)
        if backend == "pytorch_cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms.append((t1 - t0) * 1000.0)
        max_rss_mb = max(max_rss_mb, process.memory_info().rss / (1024.0 * 1024.0))

        if np.asarray(pred).ndim not in (2, 3):
            output_shape_valid = False

        label_path = _resolve_label_path(input_path, labels_dir)
        if label_path is not None and label_path.exists():
            label_arr, _ = _load_volume(str(label_path))
            if label_arr.ndim == 4 and label_arr.shape[0] == 1:
                label_arr = label_arr[0]
            if label_arr.ndim == 4 and label_arr.shape[-1] == 1:
                label_arr = label_arr[..., 0]
            if label_arr.shape == np.asarray(pred).shape:
                dice_scores.append(_foreground_dice(pred, label_arr))

    mean_ms = float(statistics.mean(latency_ms)) if latency_ms else 0.0
    return {
        "backend": backend,
        "device": "cuda" if backend == "pytorch_cuda" else "cpu",
        "precision": "fp32",
        "num_cases": len(inputs),
        "mean_latency_ms": round(mean_ms, 3),
        "p50_latency_ms": round(_percentile(latency_ms, 50), 3),
        "p95_latency_ms": round(_percentile(latency_ms, 95), 3),
        "peak_ram_mb": round(max_rss_mb, 2),
        "output_shape_valid": bool(output_shape_valid),
        "dice": round(float(statistics.mean(dice_scores)), 4) if dice_scores else None,
    }


def _write_results(results, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "backend",
        "device",
        "precision",
        "num_cases",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "peak_ram_mb",
        "model_size_mb",
        "output_shape_valid",
        "dice",
        "notes",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})

    md_path = csv_path.with_name("results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark report\n\n")
        f.write(
            "| Backend | Device | Precision | Mean latency (ms) | p95 latency (ms) | Peak RAM (MB) | Model size (MB) | Dice |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            f.write(
                "| {backend} | {device} | {precision} | {mean_latency_ms} | {p95_latency_ms} | {peak_ram_mb} | {model_size_mb} | {dice} |\n".format(
                    backend=row.get("backend"),
                    device=row.get("device"),
                    precision=row.get("precision"),
                    mean_latency_ms=row.get("mean_latency_ms"),
                    p95_latency_ms=row.get("p95_latency_ms"),
                    peak_ram_mb=row.get("peak_ram_mb"),
                    model_size_mb=row.get("model_size_mb"),
                    dice=row.get("dice") if row.get("dice") is not None else "n/a",
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark segmentation inference backends"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config .py override"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="PyTorch checkpoint path"
    )
    parser.add_argument(
        "--onnx_model", type=str, default=None, help="Optional ONNX model path"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory with inference inputs"
    )
    parser.add_argument(
        "--labels_dir", type=str, default=None, help="Optional labels directory"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch_cpu"],
        choices=["pytorch_cpu", "pytorch_cuda", "onnx_cpu", "api_cpu"],
    )
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8080")
    args = parser.parse_args()

    inputs = _discover_inputs(args.input_dir, max_cases=args.max_cases)
    config_override = _load_config_file(args.config) if args.config else {}

    model_cpu, merged_config = _build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device="cpu",
        config_override=config_override,
    )

    onnx_path = args.onnx_model
    if "onnx_cpu" in args.backends and onnx_path is None:
        output_csv = Path(args.output)
        onnx_path = str(output_csv.parent / "benchmark_model.onnx")
        ok = export_to_onnx(
            model=model_cpu,
            export_path=onnx_path,
            in_channels=int(merged_config["in_channels"]),
            input_shape=tuple(merged_config["input_shape"]),
            dynamic_axes=True,
        )
        if not ok:
            raise RuntimeError("Failed to export ONNX model for onnx_cpu backend")

    env = _collect_environment()
    env_path = Path(args.output).with_name("environment.json")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    results = []
    for backend in args.backends:
        if backend == "pytorch_cpu":
            row = _benchmark_backend(
                backend=backend,
                inputs=inputs,
                labels_dir=args.labels_dir,
                warmup=args.warmup,
                device="cpu",
                model=model_cpu,
                config=merged_config,
            )
            row["model_size_mb"] = _to_model_size_mb(args.checkpoint)
            row["notes"] = "PyTorch CPU baseline"
            results.append(row)
            continue

        if backend == "pytorch_cuda":
            if not torch.cuda.is_available():
                results.append(
                    {
                        "backend": backend,
                        "device": "cuda",
                        "precision": "fp32",
                        "num_cases": 0,
                        "mean_latency_ms": 0.0,
                        "p50_latency_ms": 0.0,
                        "p95_latency_ms": 0.0,
                        "peak_ram_mb": 0.0,
                        "model_size_mb": _to_model_size_mb(args.checkpoint),
                        "output_shape_valid": False,
                        "dice": None,
                        "notes": "CUDA not available",
                    }
                )
                continue

            model_cuda, _ = _build_model_from_checkpoint(
                checkpoint_path=args.checkpoint,
                device="cuda",
                config_override=config_override,
            )
            row = _benchmark_backend(
                backend=backend,
                inputs=inputs,
                labels_dir=args.labels_dir,
                warmup=args.warmup,
                device="cuda",
                model=model_cuda,
                config=merged_config,
            )
            row["model_size_mb"] = _to_model_size_mb(args.checkpoint)
            row["notes"] = "PyTorch CUDA"
            results.append(row)
            continue

        if backend == "onnx_cpu":
            import onnxruntime as ort

            session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
            row = _benchmark_backend(
                backend=backend,
                inputs=inputs,
                labels_dir=args.labels_dir,
                warmup=args.warmup,
                device="cpu",
                session=session,
                input_name=input_name,
            )
            row["model_size_mb"] = _to_model_size_mb(onnx_path)
            row["notes"] = "ONNX Runtime CPU"
            results.append(row)
            continue

        if backend == "api_cpu":
            row = {
                "backend": backend,
                "device": "cpu",
                "precision": "fp32",
                "num_cases": 0,
                "mean_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "peak_ram_mb": 0.0,
                "model_size_mb": _to_model_size_mb(args.checkpoint),
                "output_shape_valid": False,
                "dice": None,
                "notes": "API benchmarking placeholder. Run HTTP benchmark client against /predict.",
            }
            results.append(row)
            continue

    _write_results(results, args.output)

    print(f"Saved benchmark CSV: {args.output}")
    print(f"Saved benchmark markdown: {Path(args.output).with_name('results.md')}")
    print(f"Saved environment metadata: {env_path}")


if __name__ == "__main__":
    main()
