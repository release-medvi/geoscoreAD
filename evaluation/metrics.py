import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc
from skimage.measure import label
import pandas as pd


# --------------------------------------------------
# Resize prediction to GT resolution
# --------------------------------------------------
def _resize_to_gt(pred_map, gt_map):
    if pred_map.shape[-2:] != gt_map.shape[-2:]:
        pred_map = F.interpolate(
            pred_map,
            size=gt_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return pred_map


def _resize_to_token(pred_map, gt_map):
    """
    Resize prediction map to GT resolution.
    pred_map: [B,1,Hs,Ws]
    gt_map:   [B,Hgt,Wgt]
    """
    if pred_map.shape[-2:] != gt_map.shape[-2:]:
        gt_map = F.interpolate(
            gt_map,
            size=pred_map.shape[-2:],
            mode="nearest",
        )
    return gt_map

# --------------------------------------------------
# Flatten maps for pixel-wise metrics
# --------------------------------------------------
def _flatten_maps(pred_map, gt_map):
    pred = pred_map.reshape(-1).cpu().numpy()
    gt = (gt_map > 0).long().reshape(-1).cpu().numpy()
    return pred, gt


# --------------------------------------------------
# PRO at fixed threshold
# --------------------------------------------------
def compute_pro_at_threshold(pred_maps, gt_maps, threshold):
    pro_scores = []

    for pred, gt in zip(pred_maps, gt_maps):
        pred_bin = (pred >= threshold).astype(np.uint8)
        gt_bin = (gt > 0).astype(np.uint8)

        labeled_gt = label(gt_bin, connectivity=1)
        num_regions = labeled_gt.max()

        if num_regions == 0:
            continue

        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_gt == region_id)
            region_area = region_mask.sum()
            if region_area == 0:
                continue

            overlap = (pred_bin & region_mask).sum()
            pro_scores.append(overlap / (region_area + 1e-8))

    if len(pro_scores) == 0:
        return 0.0

    return float(np.mean(pro_scores))


# --------------------------------------------------
# PRO curve
# --------------------------------------------------
def compute_pro_curve(pred_maps, gt_maps, num_steps=100):
    pred_maps = pred_maps.detach().cpu().numpy()
    gt_maps = gt_maps.detach().cpu().numpy()

    flat_scores = pred_maps.flatten()
    thresholds = np.quantile(flat_scores, np.linspace(0, 1, num_steps))
    thresholds = np.unique(thresholds)

    fpr_values = []
    pro_values = []

    for thr in thresholds:
        pred_bin = (pred_maps >= thr).astype(np.uint8)
        gt_bin = (gt_maps > 0).astype(np.uint8)

        # PRO
        pro_scores = []
        for pred, gt in zip(pred_bin, gt_bin):
            labeled_gt = label(gt, connectivity=1)
            num_regions = labeled_gt.max()

            if num_regions == 0:
                continue

            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_gt == region_id)
                region_area = region_mask.sum()
                if region_area == 0:
                    continue

                overlap = (pred & region_mask).sum()
                pro_scores.append(overlap / (region_area + 1e-8))

        if len(pro_scores) == 0:
            continue

        pro = float(np.mean(pro_scores))

        # FPR
        fp = ((pred_bin == 1) & (gt_bin == 0)).sum()
        tn = ((pred_bin == 0) & (gt_bin == 0)).sum()
        fpr = float(fp / (fp + tn + 1e-8))

        pro_values.append(pro)
        fpr_values.append(fpr)

    return fpr_values, pro_values


# --------------------------------------------------
# Main metric computation
# --------------------------------------------------
def compute_metrics(
    stats_df, 
    map_key,
    stats_normal,
    stats_anomalous,
    calibrated_threshold,
):

    # Concatenate
    pred = torch.cat(
        [stats_normal[map_key], stats_anomalous[map_key]], dim=0
    )
    gt = torch.cat(
        [stats_normal["labels"], stats_anomalous["labels"]], dim=0
    )

    print("shapes - pred:", pred.shape, "gt:", gt.shape)
    # Resize to GT resolution
    #pred = _resize_to_gt(pred, gt)
    gt = _resize_to_token(pred, gt) 

    pred_maps = pred.squeeze(1).cpu().numpy()
    gt_maps = gt.squeeze(1).cpu().numpy()

    pred_np, gt_np = _flatten_maps(pred, gt)

    # -------------------------
    # AUROC
    # -------------------------
    stats_df.loc[map_key, "auroc"] = roc_auc_score(gt_np, pred_np)

    # -------------------------
    # Fixed threshold metrics
    # -------------------------
    thr = calibrated_threshold
    preds_bin = (pred_np >= thr).astype(np.int32)

    tp = ((preds_bin == 1) & (gt_np == 1)).sum()
    tn = ((preds_bin == 0) & (gt_np == 0)).sum()
    fp = ((preds_bin == 1) & (gt_np == 0)).sum()
    fn = ((preds_bin == 0) & (gt_np == 1)).sum()

    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    balacc = 0.5 * (recall + tn / (tn + fp + 1e-8))

    stats_df.loc[map_key, "recall"] = recall
    stats_df.loc[map_key, "precision"] = precision
    stats_df.loc[map_key, "f1"] = f1
    stats_df.loc[map_key, "balacc"] = balacc

    # -------------------------
    # PRO + AUPRO
    # -------------------------
    pro = compute_pro_at_threshold(pred_maps, gt_maps, thr)
    stats_df.loc[map_key, "pro"] = pro

    fpr_values, pro_values = compute_pro_curve(pred, gt)

    fpr = np.array(fpr_values)
    pro_curve = np.array(pro_values)

    order = np.argsort(fpr)
    fpr = fpr[order]
    pro_curve = pro_curve[order]

    unique_fpr, idx = np.unique(fpr, return_index=True)
    fpr = unique_fpr
    pro_curve = pro_curve[idx]

    stats_df.loc[map_key, "aupro"] = auc(fpr, pro_curve)

    return stats_df
