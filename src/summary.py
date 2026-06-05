"""Segmentation summary utilities.

Deterministic structured summary generation plus optional narrative rewriting.
"""

import json
import os

import numpy as np
from scipy import ndimage


def _normalize_prediction_to_label_map(prediction):
    """Convert model prediction tensor/array to a label map [D, H, W] or [H, W]."""
    pred = np.asarray(prediction)
    if pred.ndim == 0:
        raise ValueError("prediction must have at least 1 dimension")

    # If logits/probabilities are provided in channel-first format, collapse channels.
    if pred.ndim >= 3 and pred.shape[0] > 1 and np.issubdtype(pred.dtype, np.floating):
        pred = pred.argmax(axis=0)

    # If a leading singleton batch/channel remains, squeeze it.
    while pred.ndim > 0 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim not in (2, 3):
        raise ValueError(
            f"prediction must resolve to 2D or 3D label map, got shape {pred.shape}"
        )

    return pred.astype(np.int32, copy=False)


def _safe_class_name(class_names, idx):
    if class_names is None:
        return f"class_{idx}"
    if idx < len(class_names):
        return str(class_names[idx])
    return f"class_{idx}"


def generate_structured_summary(prediction, class_names, voxel_spacing) -> dict:
    """Generate deterministic segmentation summary statistics.

    Computes per-class voxel volume, physical volume, connected components,
    and largest component volume. Class index 0 is treated as background.
    """
    label_map = _normalize_prediction_to_label_map(prediction)
    spacing = tuple(float(v) for v in voxel_spacing)
    if len(spacing) != label_map.ndim:
        raise ValueError(
            f"voxel_spacing rank {len(spacing)} does not match label rank {label_map.ndim}"
        )

    voxel_volume_mm3 = float(np.prod(spacing))
    present_classes = sorted(int(v) for v in np.unique(label_map))
    max_class_idx = max(max(present_classes), len(class_names or []) - 1)

    class_stats = []
    foreground_voxels_total = 0
    for class_idx in range(1, max_class_idx + 1):
        class_mask = label_map == class_idx
        voxel_count = int(class_mask.sum())
        volume_mm3 = float(voxel_count * voxel_volume_mm3)

        if voxel_count > 0:
            cc_map, cc_count = ndimage.label(class_mask)
            component_sizes = np.bincount(cc_map.ravel())[1:]
            largest_component_voxels = (
                int(component_sizes.max()) if component_sizes.size else 0
            )
        else:
            cc_count = 0
            largest_component_voxels = 0

        largest_component_mm3 = float(largest_component_voxels * voxel_volume_mm3)
        foreground_voxels_total += voxel_count
        class_stats.append(
            {
                "class_index": class_idx,
                "class_name": _safe_class_name(class_names, class_idx),
                "voxels": voxel_count,
                "volume_mm3": volume_mm3,
                "connected_components": int(cc_count),
                "largest_component_voxels": largest_component_voxels,
                "largest_component_mm3": largest_component_mm3,
            }
        )

    warnings = []
    if foreground_voxels_total == 0:
        warnings.append("Segmentation is empty (no foreground labels > 0).")

    return {
        "shape": list(label_map.shape),
        "voxel_spacing": list(spacing),
        "voxel_volume_mm3": voxel_volume_mm3,
        "foreground_voxels": int(foreground_voxels_total),
        "classes": class_stats,
        "warnings": warnings,
    }


def generate_narrative_report(summary: dict, backend: str = "transformers") -> str:
    """Generate a human-readable report from a deterministic summary.

    backend="transformers" attempts an optional local LLM rewrite when installed.
    Any failure gracefully falls back to deterministic template text.
    """
    class_lines = []
    for item in summary.get("classes", []):
        class_lines.append(
            "- {name}: {voxels} voxels ({mm3:.2f} mm^3), {cc} connected components, "
            "largest component {largest_mm3:.2f} mm^3".format(
                name=item.get("class_name", f"class_{item.get('class_index', '?')}"),
                voxels=int(item.get("voxels", 0)),
                mm3=float(item.get("volume_mm3", 0.0)),
                cc=int(item.get("connected_components", 0)),
                largest_mm3=float(item.get("largest_component_mm3", 0.0)),
            )
        )

    deterministic_text = "\n".join(
        [
            "Segmentation summary:",
            f"- Output shape: {summary.get('shape', [])}",
            f"- Foreground voxels: {int(summary.get('foreground_voxels', 0))}",
            *class_lines,
        ]
        + [f"- Warning: {w}" for w in summary.get("warnings", [])]
    )

    if backend != "transformers":
        return deterministic_text

    try:
        from transformers import pipeline  # type: ignore

        generator = pipeline("text2text-generation", model="google/flan-t5-small")
        prompt = (
            "Rewrite the following segmentation summary into concise radiology-style "
            "observational prose. Do not provide diagnosis.\n\n"
            f"{deterministic_text}"
        )
        out = generator(prompt, max_new_tokens=180, do_sample=False)
        if out and isinstance(out, list) and "generated_text" in out[0]:
            return str(out[0]["generated_text"]).strip()
    except Exception:
        pass

    return deterministic_text


def save_structured_summary(
    summary: dict, output_json: str, output_md: str | None = None
):
    """Persist structured summary to JSON and optional markdown."""
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if output_md is not None:
        lines = [
            "# Segmentation Summary",
            "",
            f"- Shape: {summary.get('shape', [])}",
            f"- Voxel spacing: {summary.get('voxel_spacing', [])}",
            f"- Foreground voxels: {summary.get('foreground_voxels', 0)}",
            "",
            "## Classes",
            "",
        ]
        for item in summary.get("classes", []):
            lines.append(
                "- {name}: {voxels} voxels ({mm3:.2f} mm^3), {cc} connected components, largest component {largest_mm3:.2f} mm^3".format(
                    name=item.get(
                        "class_name", f"class_{item.get('class_index', '?')}"
                    ),
                    voxels=int(item.get("voxels", 0)),
                    mm3=float(item.get("volume_mm3", 0.0)),
                    cc=int(item.get("connected_components", 0)),
                    largest_mm3=float(item.get("largest_component_mm3", 0.0)),
                )
            )
        for warning in summary.get("warnings", []):
            lines.append(f"- Warning: {warning}")

        os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
