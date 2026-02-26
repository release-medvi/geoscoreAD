
import torch
from torch.nn import functional as F
from tqdm import tqdm

from scipy.optimize import minimize
import numpy as np
from scipy.special import expit

import torch
from torch.nn import functional as F


def evaluate_fewshot(
    dataloader,
    embedding_model,
    classifier,
    device,
    cfg,
):

    stats = {
        "anomalyclip_map": [],
        "fewshot_semantic_map": [],
        "labels": [],
    }

    embedding_model.eval()
    classifier.eval()

    text_features = embedding_model.text_features

    with torch.no_grad():
        for image, labels, _ in dataloader:

            image = image.to(device)
            image, grid_size = embedding_model.crop_image(image)

            feats_list = embedding_model.extract_features(image)

            # ----------------------------------------
            # Compute semantic few-shot maps
            # ----------------------------------------
            scores = compute_semantic_fewshot_maps(
                feats_list, classifier, text_features, device
            )

            N, HW = scores["anomalyclip_map"].shape
            H, W = grid_size

            for k in ["anomalyclip_map", "fewshot_semantic_map"]:
                amap = scores[k].reshape(N, 1, H, W)
                stats[k].append(amap.cpu())

            stats["labels"].append(labels.float().cpu())

    for k in stats:
        stats[k] = torch.cat(stats[k], dim=0)

    return stats

def compute_semantic_fewshot_maps(
    features_list,
    classifier,
    text_features,
    device,
    temperature=1.0,
):

    text_normal = F.normalize(text_features[0], dim=0).to(device)
    text_abnormal = F.normalize(text_features[1], dim=0).to(device)

    semantic_layers = []
    base_scores = []

    for feats in features_list:
        feats = F.normalize(feats, dim=-1)

        s_normal = feats @ text_normal
        s_abnormal = feats @ text_abnormal

        semantic_l = torch.stack([s_normal, s_abnormal], dim=-1)
        semantic_layers.append(semantic_l)

        probs = F.softmax(semantic_l / temperature, dim=-1)
        base_scores.append(probs[..., 1])

    # Zero-shot baseline
    base_score = torch.stack(base_scores, dim=0).mean(dim=0)

    # Few-shot MLP
    semantic = torch.cat(semantic_layers, dim=-1)
    N, HW, D = semantic.shape
    X = semantic.reshape(-1, D)

    logits = classifier(X.float())
    probs = torch.sigmoid(logits)
    fewshot_score = probs.reshape(N, HW)

    return {
        "anomalyclip_map": base_score,
        "fewshot_semantic_map": fewshot_score,
    }


def evaluate_fewshot_adaclip(
    dataloader,
    adaclip_model,
    classifier,
    device,
    cfg,
):

    stats = {
        "adaclip_map": [],
        "fewshot_semantic_map": [],
        "labels": [],
    }

    adaclip_model.eval()
    classifier.eval()

    with torch.no_grad():
        for image, labels, _, cls_name in dataloader:

            image = image.to(device)
            grid_size = image.shape[2] // cfg.embedding.patch_size

            # -------------------------
            # Zero-shot AdaCLIP
            # -------------------------
            amap_org, _ = adaclip_model.clip_model(
                image, cls_name, aggregation=True
            )

            amap_org = F.adaptive_avg_pool2d(
                amap_org.squeeze(1),
                (grid_size, grid_size),
            )

            stats["adaclip_map"].append(
                amap_org.unsqueeze(1).cpu()
            )

            # -------------------------
            # Multi-layer features
            # -------------------------
            maps_list, _ = adaclip_model.clip_model(
                image, cls_name, aggregation=False
            )

            pooled_maps = [
                F.adaptive_avg_pool2d(m, (grid_size, grid_size))
                for m in maps_list
            ]

            maps = torch.stack(pooled_maps, dim=1)

            N, L, C, H, W = maps.shape
            semantic = maps.permute(0, 3, 4, 1, 2).reshape(N * H * W, L * 2)

            logits = classifier(semantic.float())
            probs = torch.sigmoid(logits)
            fewshot_map = probs.reshape(N, 1, H, W)

            stats["fewshot_semantic_map"].append(fewshot_map.cpu())
            #stats["labels"].append(labels.unsqueeze(1).float().cpu())
            stats["labels"].append(labels.float().cpu())

    for k in stats:
        stats[k] = torch.cat(stats[k], dim=0)

    return stats