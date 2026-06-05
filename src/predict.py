"""Batch prediction entrypoint for medical segmentation.

Supports PyTorch checkpoints and ONNX Runtime models with the same CLI.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from eval import _pad_batch_to_window, _sliding_window_inference
from model import UNet
from summary import (
    generate_narrative_report,
    generate_structured_summary,
    save_structured_summary,
)


def _load_config_file(config_path):
    spec = importlib.util.spec_from_file_location("predict_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return {
        name: getattr(config_module, name)
        for name in dir(config_module)
        if not name.startswith("_")
    }


def _load_volume(path):
    path = str(path)
    if path.endswith(".npy"):
        arr = np.asarray(np.load(path), dtype=np.float32)
        meta = {
            "format": "npy",
            "affine": np.eye(4, dtype=np.float32),
            "voxel_spacing": tuple([1.0] * max(arr.ndim - 1, 2)),
        }
        return arr, meta

    if path.endswith(".nii") or path.endswith(".nii.gz"):
        import nibabel as nib

        nii = nib.load(path)
        arr = np.asarray(nii.get_fdata(), dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] <= 8:
            arr = np.transpose(arr, (3, 0, 1, 2))
        meta = {
            "format": "nifti",
            "affine": nii.affine,
            "voxel_spacing": tuple(
                float(v) for v in nii.header.get_zooms()[: arr.ndim]
            ),
        }
        return arr, meta

    raise ValueError(
        f"Unsupported input format for {path}. Expected .npy, .nii, or .nii.gz"
    )


def _save_label_map(path, label_map, affine=None):
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.endswith(".npy"):
        np.save(path, label_map.astype(np.uint8))
        return

    if path.endswith(".nii") or path.endswith(".nii.gz"):
        import nibabel as nib

        affine = np.eye(4, dtype=np.float32) if affine is None else affine
        nib.Nifti1Image(label_map.astype(np.uint8), affine).to_filename(path)
        return

    raise ValueError(
        f"Unsupported output format for {path}. Expected .npy, .nii, or .nii.gz"
    )


def _default_metadata_path(output_path):
    p = Path(output_path)
    if p.suffix == ".gz" and p.name.endswith(".nii.gz"):
        stem = p.name[: -len(".nii.gz")]
        return str(p.with_name(stem + ".json"))
    return str(p.with_suffix(".json"))


def _build_model_from_checkpoint(checkpoint_path, device, config_override=None):
    payload = torch.load(checkpoint_path, map_location="cpu")

    ckpt_config = payload.get("config", {}) if isinstance(payload, dict) else {}
    merged_config = dict(ckpt_config)
    if config_override:
        merged_config.update(config_override)

    required = [
        "input_shape",
        "in_channels",
        "out_channels",
        "num_stages",
        "base_chs",
        "norm_type",
        "act_type",
        "dropout",
        "norm_groups",
        "deep_supervision",
        "slice_mode",
    ]
    missing = [key for key in required if key not in merged_config]
    if missing:
        raise ValueError(
            f"Missing model configuration keys: {missing}. "
            "Provide a checkpoint with embedded config or pass --config."
        )

    model = UNet(
        input_shape=tuple(merged_config["input_shape"]),
        in_channels=int(merged_config["in_channels"]),
        out_channels=int(merged_config["out_channels"]),
        num_stages=int(merged_config["num_stages"]),
        base_chs=int(merged_config["base_chs"]),
        norm_type=merged_config["norm_type"],
        act_type=merged_config["act_type"],
        dropout=float(merged_config["dropout"]),
        norm_groups=int(merged_config["norm_groups"]),
        deep_supervision=bool(merged_config["deep_supervision"]),
    ).to(device)

    if isinstance(payload, dict) and "model" in payload:
        state = {k.removeprefix("_orig_mod."): v for k, v in payload["model"].items()}
    else:
        state = {k.removeprefix("_orig_mod."): v for k, v in payload.items()}
    model.load_state_dict(state)
    model.eval()

    return model, merged_config


def _predict_pytorch(model, volume_chw_or_chwd, config, device):
    input_shape = tuple(int(v) for v in config["input_shape"])
    out_channels = int(config["out_channels"])
    slice_mode = str(config.get("slice_mode", "fullres"))

    image_np = np.asarray(volume_chw_or_chwd, dtype=np.float32)
    if image_np.ndim == 3:
        image_np = image_np[None, ...]

    if len(input_shape) == 3:
        batch = torch.from_numpy(image_np).unsqueeze(0).to(device)
        batch, crop_slices = _pad_batch_to_window(batch, input_shape)
        probs = _sliding_window_inference(
            models=[model],
            batch=batch,
            window_size=input_shape,
            out_channels=out_channels,
            overlap=0.5,
            mirror_axes=tuple(range(len(input_shape))),
        )
        probs = probs[(slice(None), slice(None), *crop_slices)]
        preds = probs.argmax(dim=1).squeeze(0).detach().cpu().numpy()
        return preds.astype(np.int32)

    if len(input_shape) == 2:
        if slice_mode not in {"axi", "cor", "sag"}:
            raise ValueError("2D model requires slice_mode in {'axi','cor','sag'}")

        axis_by_mode = {"sag": 3, "cor": 2, "axi": 1}
        slice_axis = axis_by_mode[slice_mode]
        pred_probs = torch.zeros(
            (out_channels, *image_np.shape[1:]),
            device=device,
            dtype=torch.float32,
        )

        for slice_idx in range(image_np.shape[slice_axis]):
            image_slicer = [slice(None)] * 4
            image_slicer[slice_axis] = slice_idx
            slice_np = np.ascontiguousarray(image_np[tuple(image_slicer)])
            slice_2d = torch.from_numpy(slice_np).unsqueeze(0).to(device)

            slice_2d, crop_slices = _pad_batch_to_window(slice_2d, input_shape)
            probs_2d = _sliding_window_inference(
                models=[model],
                batch=slice_2d,
                window_size=input_shape,
                out_channels=out_channels,
                overlap=0.5,
                mirror_axes=(0, 1),
            )
            probs_2d = probs_2d[(slice(None), slice(None), *crop_slices)].squeeze(0)

            pred_slicer = [slice(None)] * 4
            pred_slicer[slice_axis] = slice_idx
            pred_probs[tuple(pred_slicer)] = probs_2d

        preds = pred_probs.argmax(dim=0).detach().cpu().numpy()
        return preds.astype(np.int32)

    raise ValueError(f"Unsupported input_shape rank: {len(input_shape)}")


def _predict_onnxruntime(session, input_name, volume_chw_or_chwd):
    x = np.asarray(volume_chw_or_chwd, dtype=np.float32)

    # For 2D models, benchmark inputs are often full volumes (C, D, H, W).
    # Run one ONNX call per slice and re-stack to match PyTorch behavior.
    if x.ndim == 4:
        pred_slices = []
        for slice_idx in range(x.shape[1]):
            slice_x = x[:, slice_idx, :, :][None, ...]  # (1, C, H, W)
            logits = session.run(None, {input_name: slice_x})[0]
            if logits.ndim < 4:
                raise ValueError(f"Unexpected ONNX output shape {logits.shape}")
            pred_slices.append(np.argmax(logits[0], axis=0).astype(np.int32))
        return np.stack(pred_slices, axis=0)

    if x.ndim == 3:
        x = x[None, ...]  # (1, C, H, W)
    elif x.ndim != 4:
        raise ValueError(f"Unsupported ONNX input rank: {x.ndim}")

    logits = session.run(None, {input_name: x})[0]
    if logits.ndim < 4:
        raise ValueError(f"Unexpected ONNX output shape {logits.shape}")
    preds = np.argmax(logits[0], axis=0)
    return preds.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation prediction on one volume."
    )
    parser.add_argument(
        "--backend", choices=["pytorch", "onnxruntime"], default="pytorch"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config .py override"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="PyTorch checkpoint"
    )
    parser.add_argument("--model", type=str, default=None, help="ONNX model path")
    parser.add_argument(
        "--input", type=str, required=True, help="Input .npy/.nii/.nii.gz"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .npy/.nii/.nii.gz"
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--summary",
        "--report",
        dest="summary",
        type=str,
        default=None,
        help="Optional structured summary JSON path",
    )
    parser.add_argument(
        "--summary_backend",
        "--report_backend",
        dest="summary_backend",
        type=str,
        default="transformers",
        help="Narrative backend: transformers or deterministic",
    )
    args = parser.parse_args()

    config_override = _load_config_file(args.config) if args.config else {}
    input_volume, input_meta = _load_volume(args.input)

    if args.backend == "pytorch":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for backend=pytorch")
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")

        model, merged_config = _build_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=device,
            config_override=config_override,
        )
        pred_labels = _predict_pytorch(model, input_volume, merged_config, device)
        source_checkpoint = args.checkpoint
        model_name = os.path.basename(args.checkpoint)
    else:
        if args.model is None:
            raise ValueError("--model is required for backend=onnxruntime")
        import onnxruntime as ort

        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        pred_labels = _predict_onnxruntime(session, input_name, input_volume)
        merged_config = config_override
        source_checkpoint = args.model
        model_name = os.path.basename(args.model)

    _save_label_map(args.output, pred_labels, affine=input_meta.get("affine"))

    sidecar_path = _default_metadata_path(args.output)
    sidecar = {
        "backend": args.backend,
        "device": args.device,
        "input": args.input,
        "output": args.output,
        "input_shape": list(np.asarray(input_volume).shape),
        "output_shape": list(pred_labels.shape),
        "checkpoint": source_checkpoint,
        "model_name": model_name,
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    if args.summary:
        class_names = merged_config.get("class_names")
        if class_names is None and "out_channels" in merged_config:
            class_names = [
                f"class_{i}" for i in range(int(merged_config["out_channels"]))
            ]

        spacing = input_meta.get("voxel_spacing")
        if spacing is None or len(spacing) < pred_labels.ndim:
            spacing = tuple([1.0] * pred_labels.ndim)
        else:
            spacing = tuple(spacing[: pred_labels.ndim])

        summary = generate_structured_summary(
            pred_labels, class_names=class_names, voxel_spacing=spacing
        )
        narrative = generate_narrative_report(summary, backend=args.summary_backend)
        summary["narrative"] = narrative
        save_structured_summary(summary, output_json=args.summary)

    print(f"Prediction saved: {args.output}")
    print(f"Metadata saved: {sidecar_path}")
    if args.summary:
        print(f"Summary saved: {args.summary}")


if __name__ == "__main__":
    main()
