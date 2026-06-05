"""Minimal FastAPI serving wrapper for segmentation inference."""

import argparse
import os
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query

from predict import (
    _build_model_from_checkpoint,
    _load_config_file,
    _load_volume,
    _predict_onnxruntime,
    _predict_pytorch,
    _save_label_map,
)
from summary import (
    generate_narrative_report,
    generate_structured_summary,
    save_structured_summary,
)

app = FastAPI(title="MedSeg API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    state = app.state
    classes = (
        state.class_names if getattr(state, "class_names", None) is not None else []
    )
    return {
        "model_name": state.model_name,
        "backend": state.backend,
        "device": state.device,
        "classes": classes,
    }


@app.post("/predict")
async def predict_endpoint(
    payload: bytes = Body(...),
    filename: str = Query(default="input.nii.gz"),
    include_summary: bool = Query(default=False),
    include_report: bool = Query(default=False),
):
    suffix = Path(filename).suffix.lower()
    if filename.lower().endswith(".nii.gz"):
        suffix = ".nii.gz"

    if suffix not in {".npy", ".nii", ".nii.gz"}:
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Use .npy, .nii, or .nii.gz"
        )

    output_root = Path(app.state.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="medseg_api_") as tmp_dir:
        in_path = Path(tmp_dir) / f"input{suffix}"
        with open(in_path, "wb") as f:
            f.write(payload)

        volume, input_meta = _load_volume(str(in_path))

        if app.state.backend == "pytorch":
            pred = _predict_pytorch(
                app.state.model, volume, app.state.config, app.state.device
            )
        else:
            pred = _predict_onnxruntime(app.state.session, app.state.input_name, volume)

        base_name = Path(filename or "prediction").name
        if base_name.endswith(".nii.gz"):
            stem = base_name[:-7]
            pred_name = f"{stem}_pred.nii.gz"
        else:
            pred_name = f"{Path(base_name).stem}_pred.npy"

        pred_path = output_root / pred_name
        _save_label_map(str(pred_path), pred, affine=input_meta.get("affine"))

        response = {
            "output_path": str(pred_path),
            "backend": app.state.backend,
            "device": app.state.device,
            "output_shape": list(pred.shape),
        }

        include_summary_requested = include_summary or include_report
        if include_summary_requested:
            spacing = input_meta.get("voxel_spacing")
            if spacing is None or len(spacing) < pred.ndim:
                spacing = tuple([1.0] * pred.ndim)
            else:
                spacing = tuple(spacing[: pred.ndim])

            summary = generate_structured_summary(
                pred,
                class_names=app.state.class_names,
                voxel_spacing=spacing,
            )
            summary["narrative"] = generate_narrative_report(
                summary, backend=app.state.summary_backend
            )

            summary_json_path = pred_path.with_suffix("")
            summary_json_path = summary_json_path.with_name(
                summary_json_path.name + "_summary.json"
            )
            save_structured_summary(summary, output_json=str(summary_json_path))
            response["summary_path"] = str(summary_json_path)
            response["summary"] = summary
            # Backward-compatible response keys.
            response["report_path"] = str(summary_json_path)
            response["report"] = summary

        return response


def configure_app_from_args(args):
    app.state.backend = args.backend
    app.state.device = args.device
    app.state.output_dir = args.output_dir
    app.state.summary_backend = args.summary_backend

    config_override = _load_config_file(args.config) if args.config else {}

    if args.backend == "pytorch":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for backend=pytorch")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")

        model, merged_config = _build_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
            config_override=config_override,
        )
        app.state.model = model
        app.state.config = merged_config
        app.state.model_name = os.path.basename(args.checkpoint)
        app.state.class_names = merged_config.get("class_names")
        if app.state.class_names is None and "out_channels" in merged_config:
            app.state.class_names = [
                f"class_{i}" for i in range(int(merged_config["out_channels"]))
            ]
    else:
        if args.model is None:
            raise ValueError("--model is required for backend=onnxruntime")
        import onnxruntime as ort

        app.state.session = ort.InferenceSession(
            args.model, providers=["CPUExecutionProvider"]
        )
        app.state.input_name = app.state.session.get_inputs()[0].name
        app.state.model_name = os.path.basename(args.model)
        app.state.class_names = config_override.get("class_names")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve segmentation model via FastAPI."
    )
    parser.add_argument(
        "--backend", choices=["pytorch", "onnxruntime"], default="pytorch"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config .py override"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="PyTorch checkpoint path"
    )
    parser.add_argument("--model", type=str, default=None, help="ONNX model path")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output_dir", type=str, default="outputs/api")
    parser.add_argument(
        "--summary_backend",
        "--report_backend",
        dest="summary_backend",
        type=str,
        default="deterministic",
        help="Narrative backend: deterministic or transformers",
    )
    args = parser.parse_args()

    configure_app_from_args(args)
    uvicorn.run(app, host=args.host, port=args.port)
