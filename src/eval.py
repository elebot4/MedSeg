"""
Evaluation and metrics for medical segmentation models.
Supports various metrics like Dice, IoU, Hausdorff distance, and inference pipelines.
"""

import itertools
import json
import os

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

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


@torch.no_grad()
def _sliding_window_inference(model, batch, window_size, out_channels, overlap=0.25):
    model.eval()

    device = next(model.parameters()).device
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
    output = torch.zeros(size=output_shape, device=device)
    count = torch.zeros(spatial_shape, device=device)

    for idxs in itertools.product(*starts):
        slices = tuple(slice(i, i + w) for i, w in zip(idxs, window_size))
        window = batch[(slice(None), slice(None), *slices)]

        pred = model(window)
        pred = torch.softmax(pred, dim=1).squeeze(0)

        if output is None:
            # HACK: initialize output from pred. shape. In case model comes from outside with no nb_heads property
            output = torch.zeros(
                (pred.shape[0], *spatial_shape), device=device, dtype=pred.dtype
            )

        output[(slice(None), slice(None), *slices)] += pred
        count[slices] += 1

    return output / count.unsqueeze(0)


@torch.no_grad()
def run_eval(
    model,
    data_dir,
    batch_size,
    slice_mode,
    out_channels,
    save_path=None,
    do_surface=False,
):
    model.eval()
    device = next(model.parameters()).device

    images_dir = os.path.join(data_dir, "imagesTs")
    labels_dir = os.path.join(data_dir, "labelsTs")

    image_paths = sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.endswith(".nii.gz")
    )

    if save_path is not None:
        cases_dir = os.path.join(save_path, "cases")
        os.makedirs(cases_dir, exist_ok=True)

    all_dice = []
    all_iou = []
    all_nsd = []
    all_hd95 = []
    per_case = []

    print(f"Evaluating model on {len(image_paths)} cases...")

    for case_idx, image_path in enumerate(image_paths, start=1):
        filename = os.path.basename(image_path)
        case_id = filename.removesuffix(".nii.gz").removesuffix("_0000")
        gt_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")

        image_np = np.asarray(nib.load(image_path).dataobj, dtype="float32")  # ty:ignore
        gt_np = np.asarray(nib.load(gt_path).dataobj, dtype="int64")  # ty:ignore

        batch = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
        gt = torch.from_numpy(gt_np).unsqueeze(0).to(device).long()

        preds = _sliding_window_inference(
            model, batch, window_size=(64, 64, 64), out_channels=out_channels
        )
        preds = preds.argmax(dim=1)

        gt_shape = (gt.shape[0], out_channels, *gt.shape[1:])
        gt_onehot = torch.zeros(gt_shape, device=device)
        pred_onehot = torch.zeros_like(gt_onehot)

        gt_onehot.scatter_(1, gt.unsqueeze(1), 1)
        pred_onehot.scatter_(1, preds.unsqueeze(1), 1)

        dice = np.atleast_2d(dice_score(pred_onehot, gt_onehot))
        iou = np.atleast_2d(iou_score(pred_onehot, gt_onehot))

        all_dice.append(dice)
        all_iou.append(iou)

        case_metrics = {
            "case_id": case_id,
            "image_path": image_path,
            "gt_path": gt_path,
            "dice_mean": float(dice.mean()),
            "iou_mean": float(iou.mean()),
        }

        for c in range(out_channels):
            case_metrics[f"dice_class_{c}"] = float(dice[0, c])
            case_metrics[f"iou_class_{c}"] = float(iou[0, c])

        if do_surface:
            nsd = np.atleast_2d(nsd_score(pred_onehot, gt_onehot))
            hd95 = np.atleast_2d(hd95_score(pred_onehot, gt_onehot))

            all_nsd.append(nsd)
            all_hd95.append(hd95)

            case_metrics["nsd_mean"] = float(nsd.mean())
            case_metrics["hd95_mean"] = float(hd95.mean())

            for c in range(out_channels):
                case_metrics[f"nsd_class_{c}"] = float(nsd[0, c])
                case_metrics[f"hd95_class_{c}"] = float(hd95[0, c])

        per_case.append(case_metrics)

        if save_path is not None:
            case_dir = os.path.join(cases_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)

            image_link = os.path.join(case_dir, "image.nii.gz")
            gt_link = os.path.join(case_dir, "gt.nii.gz")

            if not os.path.exists(image_link):
                os.symlink(os.path.abspath(image_path), image_link)

            if not os.path.exists(gt_link):
                os.symlink(os.path.abspath(gt_path), gt_link)

            np.save(
                os.path.join(case_dir, "pred.npy"),
                preds.squeeze(0).detach().cpu().numpy(),
            )

            with open(os.path.join(case_dir, "metrics.json"), "w") as f:
                json.dump(case_metrics, f, indent=2)

        if case_idx % 10 == 0:
            print(f"Processed {case_idx}/{len(image_paths)} cases")

    dice_scores = np.concatenate(all_dice, axis=0)
    iou_scores = np.concatenate(all_iou, axis=0)

    summary = {
        "num_cases": len(image_paths),
        "dice_mean": float(dice_scores.mean()),
        "dice_std": float(dice_scores.std()),
        "iou_mean": float(iou_scores.mean()),
        "iou_std": float(iou_scores.std()),
    }

    per_class = {}

    for c in range(out_channels):
        per_class[str(c)] = {
            "dice_mean": float(dice_scores[:, c].mean()),
            "iou_mean": float(iou_scores[:, c].mean()),
        }

    if do_surface:
        nsd_scores = np.concatenate(all_nsd, axis=0)
        hd95_scores = np.concatenate(all_hd95, axis=0)

        summary.update(
            {
                "nsd_mean": float(nsd_scores.mean()),
                "nsd_std": float(nsd_scores.std()),
                "hd95_mean": float(hd95_scores.mean()),
                "hd95_std": float(hd95_scores.std()),
            }
        )

        for c in range(out_channels):
            per_class[str(c)]["nsd_mean"] = float(nsd_scores[:, c].mean())
            per_class[str(c)]["hd95_mean"] = float(hd95_scores[:, c].mean())

    results = {
        "config": {
            "data_dir": str(data_dir),
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "batch_size": batch_size,
            "slice_mode": slice_mode,
            "out_channels": out_channels,
            "do_surface": do_surface,
        },
        "summary": summary,
        "per_class": per_class,
        "per_case": per_case,
    }

    if save_path is not None:
        with open(os.path.join(save_path, "summary_metrics.json"), "w") as f:
            json.dump(
                {
                    "config": results["config"],
                    "summary": summary,
                    "per_class": per_class,
                },
                f,
                indent=2,
            )

        with open(os.path.join(save_path, "per_case_metrics.json"), "w") as f:
            json.dump(per_case, f, indent=2)

        print(f"Results saved to: {save_path}")

    return results
