"""
Evaluation and metrics for medical segmentation models.
Supports various metrics like Dice, IoU, Hausdorff distance, and inference pipelines.
"""

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from model import UNet

### --- Medical Segmentation Decathlon Challenge Metrics ---


def dice_score(mask_ref, mask_pred, eval_mask=None):
    if eval_mask is None:
        eval_mask = np.ones_like(mask_ref, dtype=bool)

    axes = tuple(range(2, mask_ref.ndim))

    inter = ((mask_ref & mask_pred) & eval_mask).sum(axis=axes)
    ref_sum = (mask_ref & eval_mask).sum(axis=axes)
    pred_sum = (mask_pred & eval_mask).sum(axis=axes)

    dice = (2 * inter + 1e-6) / (ref_sum + pred_sum + 1e-6)

    return dice


def iou_score(mask_ref, mask_pred, eval_mask=None):
    if eval_mask is None:
        eval_mask = np.ones_like(mask_ref, dtype=bool)

    axes = tuple(range(2, mask_ref.ndim))

    inter = ((mask_ref & mask_pred) & eval_mask).sum(axis=axes)
    union = ((mask_ref | mask_pred) & eval_mask).sum(axis=axes)

    iou = (inter + 1e-6) / (union + 1e-6)

    return iou


def _get_surface(mask):
    mask = mask.astype(bool, copy=False)
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded


def _nsd_score_binary(
    mask_ref,
    mask_pred,
    eval_mask=None,
    spacing=None,
    tolerance=2.0,
):
    if eval_mask is None:
        eval_mask = np.ones_like(mask_ref, dtype=bool)

    surface_ref = _get_surface(mask_ref) & eval_mask
    surface_pred = _get_surface(mask_pred) & eval_mask

    dist_to_ref = ndimage.distance_transform_edt(
        ~surface_ref,
        sampling=spacing,
    )

    dist_to_pred = ndimage.distance_transform_edt(
        ~surface_pred,
        sampling=spacing,
    )

    pred_close = dist_to_ref[surface_pred] <= tolerance
    ref_close = dist_to_pred[surface_ref] <= tolerance

    nsd = (pred_close.sum() + ref_close.sum() + 1e-6) / (
        surface_ref.sum() + surface_pred.sum() + 1e-6
    )

    return float(nsd)


def nsd_score(
    mask_ref,
    mask_pred,
    eval_mask=None,
    spacing=None,
    tolerance=2.0,
):
    """
    mask_ref:  bool array [B, C, *spatial_dims]
    mask_pred: bool array [B, C, *spatial_dims]
    eval_mask: bool array [B, C, *spatial_dims], True = evaluate
    spacing:   voxel spacing, e.g. (1.0, 1.0, 1.0)
    tolerance: surface tolerance in physical units

    returns:
        NSD score array [B, C]
    """
    B, C = mask_ref.shape[:2]
    nsd = np.empty((B, C), dtype=np.float32)

    if eval_mask is None:
        eval_mask = np.ones_like(mask_ref, dtype=bool)

    for b in range(B):
        for c in range(C):
            nsd[b, c] = _nsd_score_binary(
                mask_ref=mask_ref[b, c],
                mask_pred=mask_pred[b, c],
                eval_mask=eval_mask[b, c],
                spacing=spacing,
                tolerance=tolerance,
            )

    return nsd


### --- Brain Tumor Segmentation Challenge Additional Metrics ---


def _hd95_binary(
    mask_ref,
    mask_pred,
    eval_mask=None,
    spacing=None,
):
    if eval_mask is None:
        eval_mask = np.ones_like(mask_ref, dtype=bool)

    surface_ref = _get_surface(mask_ref) & eval_mask
    surface_pred = _get_surface(mask_pred) & eval_mask

    if surface_ref.sum() == 0 and surface_pred.sum() == 0:
        return 0.0

    if surface_ref.sum() == 0 or surface_pred.sum() == 0:
        return np.inf

    dist_to_ref = ndimage.distance_transform_edt(
        ~surface_ref,
        sampling=spacing,
    )

    dist_to_pred = ndimage.distance_transform_edt(
        ~surface_pred,
        sampling=spacing,
    )

    pred_to_ref = dist_to_ref[surface_pred]
    ref_to_pred = dist_to_pred[surface_ref]

    distances = np.concatenate([pred_to_ref, ref_to_pred])

    return float(np.percentile(distances, 95))


def hd95_score(
    mask_ref,
    mask_pred,
    eval_mask=None,
    spacing=None,
):
    """
    mask_ref:  bool array [B, C, *spatial_dims]
    mask_pred: bool array [B, C, *spatial_dims]
    eval_mask: bool array [B, C, *spatial_dims], True = evaluate
    spacing:   voxel spacing, e.g. (1.0, 1.0, 1.0)

    returns:
        HD95 array [B, C]
    """
    B, C = mask_ref.shape[:2]
    hd95 = np.empty((B, C), dtype=np.float32)

    if eval_mask is None:
        eval_mask = np.ones_like(mask_ref, dtype=bool)

    for b in range(B):
        for c in range(C):
            hd95[b, c] = _hd95_binary(
                mask_ref=mask_ref[b, c],
                mask_pred=mask_pred[b, c],
                eval_mask=eval_mask[b, c],
                spacing=spacing,
            )

    return hd95


def _propagate_last_channel_to_first_foreground(onehot: torch.Tensor) -> torch.Tensor:
    if onehot.shape[1] > 2:
        onehot[:, 1] = torch.maximum(onehot[:, 1], onehot[:, -1])
    return onehot


def _gaussian_importance_map(window_size, device):
    coords = [
        torch.arange(size, device=device, dtype=torch.float32) for size in window_size
    ]
    grids = torch.meshgrid(*coords, indexing="ij")

    gaussian = torch.ones(window_size, device=device, dtype=torch.float32)
    for grid, size in zip(grids, window_size):
        center = (size - 1) / 2.0
        sigma = max(size / 8.0, 1e-6)
        gaussian = gaussian * torch.exp(-((grid - center) ** 2) / (2 * sigma * sigma))

    gaussian = gaussian / gaussian.max().clamp_min(1e-6)
    return gaussian.clamp_min(1e-4)


def _model_logits(model, window):
    logits = model(window)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits


def _predict_with_mirroring(model, window, mirror_axes):
    spatial_dims = window.ndim - 2
    valid_axes = tuple(axis for axis in mirror_axes if 0 <= axis < spatial_dims)

    probs_sum = None
    num_predictions = 0

    for mask in range(1 << len(valid_axes)):
        axes = [valid_axes[i] for i in range(len(valid_axes)) if mask & (1 << i)]
        flip_dims = [axis + 2 for axis in axes]

        window_input = torch.flip(window, dims=flip_dims) if flip_dims else window
        probs = torch.softmax(_model_logits(model, window_input), dim=1).float()

        if flip_dims:
            probs = torch.flip(probs, dims=flip_dims)

        probs_sum = probs if probs_sum is None else probs_sum + probs
        num_predictions += 1

    return probs_sum / max(num_predictions, 1)


@torch.no_grad()
def _sliding_window_inference(
    models, batch, window_size, out_channels, overlap=0.5, mirror_axes=()
):
    for model in models:
        model.eval()

    device = next(models[0].parameters()).device
    batch = batch.to(device)
    spatial_shape = batch.shape[2:]

    if len(window_size) != len(spatial_shape):
        raise ValueError(
            f"window_size {window_size} does not match image shape {spatial_shape}"
        )

    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    steps = [max(1, int(w * (1 - overlap))) for w in window_size]

    starts = []
    for size, win, step in zip(spatial_shape, window_size, steps):
        if win > size:
            raise ValueError(
                f"window_size {window_size} larger than image shape {spatial_shape}"
            )

        s = list(range(0, size - win + 1, step))
        if s[-1] != size - win:
            s.append(size - win)
        starts.append(s)

    output_shape = (batch.shape[0], out_channels, *spatial_shape)
    output = torch.zeros(size=output_shape, device=device, dtype=torch.float32)
    count = torch.zeros(spatial_shape, device=device, dtype=torch.float32)
    importance_map = _gaussian_importance_map(window_size, device)

    for idxs in itertools.product(*starts):
        slices = tuple(slice(i, i + w) for i, w in zip(idxs, window_size))
        window = batch[(slice(None), slice(None), *slices)]

        probs = torch.zeros(
            (batch.shape[0], out_channels, *window_size),
            device=device,
            dtype=torch.float32,
        )
        for model in models:
            probs += _predict_with_mirroring(model, window, mirror_axes)
        probs /= len(models)

        output[(slice(None), slice(None), *slices)] += probs * importance_map
        count[slices] += importance_map

    norm = count.clamp_min(1e-6).unsqueeze(0).unsqueeze(0)
    return output / norm


def _pad_batch_to_window(batch, window_size):
    spatial_shape = batch.shape[2:]

    if len(window_size) != len(spatial_shape):
        raise ValueError(
            f"window_size {window_size} does not match batch shape {spatial_shape}"
        )

    pads = []
    crop_slices = []
    for size, win in zip(reversed(spatial_shape), reversed(window_size)):
        if size < win:
            pad_total = win - size
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
        else:
            pad_before = 0
            pad_after = 0

        pads.extend([pad_before, pad_after])
        crop_slices.append(slice(pad_before, pad_before + size))

    if any(pads):
        batch = F.pad(batch, pads, mode="constant", value=0)

    return batch, tuple(reversed(crop_slices))


@torch.no_grad()
def run_eval(
    models,
    data_dir,
    batch_size,
    slice_mode,
    out_channels,
    input_shape=(32, 32, 32),
    train_split=0.8,
    seed=42,
    save_path=None,
    do_surface=False,
    overlap=0.5,
    use_mirroring=True,
    checkpoint_paths=None,
):
    if not models:
        raise ValueError("run_eval requires at least one model")

    for model in models:
        model.eval()
    device = next(models[0].parameters()).device

    if out_channels <= 1:
        raise ValueError(
            f"out_channels must be > 1 to ignore background channel 0, got {out_channels}"
        )

    class_indices = list(range(1, out_channels))

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    discovered_cases = []
    image_paths = sorted(
        {path.resolve() for path in Path(data_dir).glob("**/image.npy")}
    )
    if not image_paths:
        raise ValueError(f"No image.npy files found recursively in {data_dir}")

    for image_path in image_paths:
        if not image_path.is_file():
            continue

        gt_path = Path(str(image_path).replace("image.npy", "labels.npy"))
        if not gt_path.is_file():
            raise FileNotFoundError(
                f"Missing labels file for {image_path}: expected {gt_path}"
            )

        case_rel_dir = os.path.relpath(image_path.parent, data_dir)
        case_id = image_path.parent.name
        discovered_cases.append(
            {
                "case_id": case_id,
                "case_rel_dir": case_rel_dir,
                "image_path": str(image_path),
                "gt_path": str(gt_path),
            }
        )

    if not discovered_cases:
        raise ValueError(f"No valid .npy cases found recursively in {data_dir}")

    # Match split construction behavior from dataset.get_dataloaders exactly.
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")

    required_files = ("image.npy", "labels.npy", "metadata.json")
    case_names = []

    for name in sorted(os.listdir(data_dir)):
        case_dir = os.path.join(data_dir, name)
        if not os.path.isdir(case_dir):
            continue

        missing = [
            filename
            for filename in required_files
            if not os.path.isfile(os.path.join(case_dir, filename))
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing files in {case_dir}: {missing}. "
                f"Required: {list(required_files)}"
            )

        case_names.append(name)

    if len(case_names) < 2:
        raise ValueError(
            f"Need at least 2 valid cases for train/val split, found {len(case_names)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(case_names)

    split_idx = int(len(case_names) * train_split)
    train_files_list = case_names[:split_idx]
    val_files_list = case_names[split_idx:]

    if not train_files_list or not val_files_list:
        raise ValueError(
            f"Empty train/val split with {len(case_names)} cases and train_split={train_split}"
        )

    split_info = {
        "data_dir": str(Path(data_dir).resolve()),
        "seed": int(seed),
        "train_split": float(train_split),
        "num_cases": len(case_names),
        "train_files": train_files_list,
        "val_files": val_files_list,
    }

    train_files = set(split_info["train_files"])

    cases = []
    excluded_train_cases = []
    for case in discovered_cases:
        rel_parts = case["case_rel_dir"].split(os.sep)
        top_level_dir = rel_parts[0] if rel_parts else case["case_rel_dir"]
        split_keys = {case["case_id"], case["case_rel_dir"], top_level_dir}

        if split_keys & train_files:
            excluded_train_cases.append(case["case_rel_dir"])
            continue

        cases.append(case)

    if not cases:
        raise ValueError(
            "No evaluation cases left after filtering training files from seeded split. "
            "Check data_dir/train_split/seed."
        )

    split_info["excluded_train_cases"] = sorted(set(excluded_train_cases))
    split_info["included_eval_cases"] = sorted({c["case_rel_dir"] for c in cases})
    split_info["num_discovered_cases"] = len(discovered_cases)
    split_info["num_eval_cases"] = len(cases)

    if save_path is not None:
        cases_dir = os.path.join(save_path, "cases")
        os.makedirs(cases_dir, exist_ok=True)

        with open(os.path.join(save_path, "split_used.json"), "w") as f:
            json.dump(split_info, f, indent=2)

    all_dice = []
    all_iou = []
    all_nsd = []
    all_hd95 = []
    per_case = []

    print(
        f"Evaluating model on {len(cases)} .npy cases "
        f"(filtered out {len(excluded_train_cases)} train cases)..."
    )
    eval_iter = enumerate(cases, start=1)

    for case_idx, eval_item in eval_iter:
        case_id = eval_item["case_id"]
        image_path = eval_item["image_path"]
        gt_path = eval_item["gt_path"]
        image_np = np.asarray(np.load(image_path, mmap_mode="r"), dtype="float32")
        gt_np = np.asarray(np.load(gt_path, mmap_mode="r"), dtype="int64")
        gt_np[gt_np < 0] = 0

        if image_np.ndim == 3:
            image_np = image_np[None, ...]
        elif image_np.ndim != 4:
            raise ValueError(
                f"Expected image with 3D or 4D shape for case {case_id}, got {image_np.shape}"
            )

        if gt_np.ndim == 4 and gt_np.shape[0] == 1:
            gt_np = gt_np[0]
        elif gt_np.ndim != 3:
            raise ValueError(
                f"Expected labels with 3D shape for case {case_id}, got {gt_np.shape}"
            )

        gt = torch.from_numpy(gt_np).unsqueeze(0).to(device).long()

        if len(input_shape) == 3:
            batch = torch.from_numpy(image_np).unsqueeze(0).to(device)
            window_size = tuple(int(v) for v in input_shape)
            batch, crop_slices = _pad_batch_to_window(batch, window_size)
            mirror_axes = tuple(range(len(window_size))) if use_mirroring else ()

            probs = _sliding_window_inference(
                models,
                batch,
                window_size=window_size,
                out_channels=out_channels,
                overlap=overlap,
                mirror_axes=mirror_axes,
            )
            probs = probs[(slice(None), slice(None), *crop_slices)]
            preds = probs.argmax(dim=1)
        elif len(input_shape) == 2:
            if slice_mode not in {"axi", "cor", "sag"}:
                raise ValueError(
                    "2D evaluation requires slice_mode in {'axi', 'cor', 'sag'}, "
                    f"got {slice_mode}"
                )

            axis_by_mode = {"sag": 3, "cor": 2, "axi": 1}
            slice_axis = axis_by_mode[slice_mode]
            base_window_2d = tuple(int(v) for v in input_shape)
            mirror_axes = (0, 1) if use_mirroring else ()

            pred_probs = torch.zeros(
                (out_channels, *image_np.shape[1:]),
                device=device,
                dtype=torch.float32,
            )

            for slice_idx in range(image_np.shape[slice_axis]):
                image_slicer = [slice(None)] * 4
                image_slicer[slice_axis] = slice_idx  # ty:ignore

                slice_np = np.ascontiguousarray(image_np[tuple(image_slicer)])
                slice_2d = torch.from_numpy(slice_np).unsqueeze(0).to(device)

                slice_2d, crop_slices = _pad_batch_to_window(slice_2d, base_window_2d)
                probs_2d = _sliding_window_inference(
                    models,
                    slice_2d,
                    window_size=base_window_2d,
                    out_channels=out_channels,
                    overlap=overlap,
                    mirror_axes=mirror_axes,
                )
                probs_2d = probs_2d[(slice(None), slice(None), *crop_slices)].squeeze(0)

                pred_slicer = [slice(None)] * 4
                pred_slicer[slice_axis] = slice_idx  # ty:ignore
                pred_probs[tuple(pred_slicer)] = probs_2d

            preds = pred_probs.argmax(dim=0, keepdim=True)
        else:
            raise ValueError(
                f"Unsupported input_shape dimensions: {len(input_shape)} for case {case_id}"
            )

        gt_shape = (gt.shape[0], out_channels, *gt.shape[1:])
        gt_onehot = torch.zeros(gt_shape, device=device, dtype=torch.bool)
        pred_onehot = torch.zeros_like(gt_onehot, dtype=torch.bool)

        gt_onehot.scatter_(1, gt.unsqueeze(1), 1)
        pred_onehot.scatter_(1, preds.unsqueeze(1), 1)
        gt_onehot = _propagate_last_channel_to_first_foreground(gt_onehot)
        pred_onehot = _propagate_last_channel_to_first_foreground(pred_onehot)

        gt_onehot_np = gt_onehot.detach().cpu().numpy()
        pred_onehot_np = pred_onehot.detach().cpu().numpy()

        dice = np.atleast_2d(dice_score(pred_onehot_np, gt_onehot_np))
        iou = np.atleast_2d(iou_score(pred_onehot_np, gt_onehot_np))

        all_dice.append(dice)
        all_iou.append(iou)

        case_metrics = {
            "case_id": case_id,
            "image_path": image_path,
            "gt_path": gt_path,
            "dice_mean": float(dice[:, class_indices].mean()),
            # "iou_mean": float(iou[:, class_indices].mean()),
        }

        for c in class_indices:
            case_metrics[f"dice_class_{c}"] = float(dice[0, c])
            case_metrics[f"iou_class_{c}"] = float(iou[0, c])

        if do_surface:
            nsd = np.atleast_2d(nsd_score(pred_onehot_np, gt_onehot_np))
            hd95 = np.atleast_2d(hd95_score(pred_onehot_np, gt_onehot_np))

            all_nsd.append(nsd)
            all_hd95.append(hd95)

            case_metrics["nsd_mean"] = float(nsd[:, class_indices].mean())
            case_metrics["hd95_mean"] = float(hd95[:, class_indices].mean())

            for c in class_indices:
                case_metrics[f"nsd_class_{c}"] = float(nsd[0, c])
                case_metrics[f"hd95_class_{c}"] = float(hd95[0, c])

        per_case.append(case_metrics)

        if save_path is not None:
            case_dir = os.path.join(cases_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)

            # nib.Nifti1Image(
            #    preds.squeeze(0).detach().cpu().numpy().astype(np.uint8), np.eye(4)
            # ).to_filename(os.path.join(case_dir, "pred.nii.gz"))

            with open(os.path.join(case_dir, "metrics.json"), "w") as f:
                json.dump(case_metrics, f, indent=2)

        if case_idx % 10 == 0:
            total_cases = len(cases)
            print(f"Processed {case_idx}/{total_cases} cases")

    dice_scores = np.concatenate(all_dice, axis=0)
    iou_scores = np.concatenate(all_iou, axis=0)

    summary = {
        "num_cases": int(dice_scores.shape[0]),
        "dice_mean": float(dice_scores[:, class_indices].mean()),
        "dice_std": float(dice_scores[:, class_indices].std()),
        # "iou_mean": float(iou_scores[:, class_indices].mean()),
        # "iou_std": float(iou_scores[:, class_indices].std()),
    }

    per_class = {}

    for c in class_indices:
        per_class[str(c)] = {
            "dice_mean": float(dice_scores[:, c].mean()),
            "iou_mean": float(iou_scores[:, c].mean()),
        }

    if do_surface:
        nsd_scores = np.concatenate(all_nsd, axis=0)
        hd95_scores = np.concatenate(all_hd95, axis=0)

        summary.update(
            {
                "nsd_mean": float(nsd_scores[:, class_indices].mean()),
                "nsd_std": float(nsd_scores[:, class_indices].std()),
                "hd95_mean": float(hd95_scores[:, class_indices].mean()),
                "hd95_std": float(hd95_scores[:, class_indices].std()),
            }
        )

        for c in class_indices:
            per_class[str(c)]["nsd_mean"] = float(nsd_scores[:, c].mean())
            per_class[str(c)]["hd95_mean"] = float(hd95_scores[:, c].mean())

    results = {
        "config": {
            "data_dir": str(data_dir),
            "batch_size": batch_size,
            "slice_mode": slice_mode,
            "out_channels": out_channels,
            "input_shape": input_shape,
            "train_split": train_split,
            "seed": seed,
            "eval_source": "recursive_npy_files",
            "split_file": "split_used.json" if save_path is not None else None,
            "do_surface": do_surface,
            "overlap": overlap,
            "use_mirroring": use_mirroring,
            "checkpoints": checkpoint_paths or [],
        },
        "summary": summary,
        "per_class": per_class,
        "per_case": per_case,
        "split": split_info,
    }

    if save_path is not None:
        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump(
                {
                    "config": results["config"],
                    "summary": summary,
                    "per_class": per_class,
                },
                f,
                indent=2,
            )

        with open(os.path.join(save_path, "per_case.json"), "w") as f:
            json.dump(per_case, f, indent=2)

        print(f"Results saved to: {save_path}")

    return results


def _load_config_file(config_path):
    spec = importlib.util.spec_from_file_location("eval_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return {
        name: getattr(config_module, name)
        for name in dir(config_module)
        if not name.startswith("_")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run segmentation model evaluation on recursively discovered .npy files."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config .py file"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Run directory containing .checkpoints/checkpoint_final.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit checkpoint path(s) to ensemble (override --eval_dir default).",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Evaluation data directory"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory to save evaluation outputs (default: --eval_dir)",
    )

    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--train_split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--disable_mirroring",
        action="store_true",
        help="Disable flip-based test-time augmentation.",
    )
    parser.add_argument(
        "--do_surface", action="store_true", help="Compute NSD and HD95"
    )

    args = parser.parse_args()

    # Optional eval-time config file overrides.
    config = {}
    if args.config is not None:
        config = _load_config_file(args.config)

    eval_dir = os.path.abspath(args.eval_dir)
    checkpoint_paths = (
        [os.path.abspath(path) for path in args.checkpoint]
        if args.checkpoint is not None
        else [os.path.join(eval_dir, ".checkpoints", "checkpoint_final.pt")]
    )

    for checkpoint_path in checkpoint_paths:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Provide --checkpoint or ensure .checkpoints/checkpoint_final.pt exists under --eval_dir."
            )

    checkpoint_payloads = [
        torch.load(path, map_location="cpu") for path in checkpoint_paths
    ]
    ckpt_config = checkpoint_payloads[0]["config"]

    device = (
        args.device
        if args.device is not None
        else config.get(
            "device",
            ckpt_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
    )
    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else config.get(
            "data_dir", ckpt_config.get("data_dir", "data/processed/Task01_BrainTumour")
        )
    )
    batch_size = int(config.get("batch_size", ckpt_config["batch_size"]))
    slice_mode = config.get("slice_mode", ckpt_config["slice_mode"])
    train_split = float(
        args.train_split
        if args.train_split is not None
        else config.get("train_split", ckpt_config.get("train_split", 0.8))
    )
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("seed", ckpt_config.get("seed", 42))
    )

    save_path = args.save_path
    if save_path is None:
        save_path = eval_dir
    os.makedirs(save_path, exist_ok=True)

    input_shape = tuple(config.get("input_shape", ckpt_config["input_shape"]))
    norm_type = config.get("norm_type", ckpt_config["norm_type"])
    act_type = config.get("act_type", ckpt_config["act_type"])
    dropout = float(config.get("dropout", ckpt_config["dropout"]))
    norm_groups = int(config.get("norm_groups", ckpt_config["norm_groups"]))
    deep_supervision = bool(
        config.get("deep_supervision", ckpt_config["deep_supervision"])
    )

    model_kwargs = {
        "input_shape": input_shape,
        "in_channels": int(config.get("in_channels", ckpt_config["in_channels"])),
        "out_channels": int(config.get("out_channels", ckpt_config["out_channels"])),
        "num_stages": int(config.get("num_stages", ckpt_config["num_stages"])),
        "base_chs": int(config.get("base_chs", ckpt_config["base_chs"])),
        "norm_type": norm_type,
        "act_type": act_type,
        "dropout": dropout,
        "norm_groups": norm_groups,
        "deep_supervision": deep_supervision,
    }

    print(f"Loading model on {device}...")
    print(
        "Model config: "
        f"in_channels={model_kwargs['in_channels']}, "
        f"out_channels={model_kwargs['out_channels']}, "
        f"num_stages={model_kwargs['num_stages']}, "
        f"base_chs={model_kwargs['base_chs']}, "
        f"norm_type={model_kwargs['norm_type']}"
    )
    models = []
    compare_keys = [
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

    for checkpoint_path, checkpoint_payload in zip(
        checkpoint_paths, checkpoint_payloads
    ):
        other_config = checkpoint_payload["config"]
        for key in compare_keys:
            if other_config.get(key) != ckpt_config.get(key):
                raise ValueError(
                    f"Checkpoint config mismatch for '{key}': {checkpoint_path} has {other_config.get(key)!r}, expected {ckpt_config.get(key)!r}"
                )

        state = {
            k.removeprefix("_orig_mod."): v
            for k, v in checkpoint_payload["model"].items()
        }
        model = UNet(**model_kwargs).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} checkpoint(s) for inference")

    results = run_eval(
        models=models,
        data_dir=data_dir,
        batch_size=batch_size,
        slice_mode=slice_mode,
        out_channels=model_kwargs["out_channels"],
        input_shape=input_shape,
        train_split=train_split,
        seed=seed,
        save_path=save_path,
        do_surface=args.do_surface,
        overlap=args.overlap,
        use_mirroring=not args.disable_mirroring,
        checkpoint_paths=checkpoint_paths,
    )

    print("Evaluation complete.")
    print(json.dumps(results["summary"], indent=2))
