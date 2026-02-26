import os
import random
import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset

from data.data import get_datasets
from training.fewshot_trainer import train_few_shot_classifier
from utils.utils import load_config, makedirs, setup_device
from evaluation.test import test_fewshot_model
from evaluation.evaluate_clip import evaluate_fewshot, evaluate_fewshot_adaclip

from external.AnomalyCLIP_lib.inference import AnomalyCLIP_Inference
from external.AdaCLIP import AdaCLIP_Inference


# --------------------------------------------------
# Args
# --------------------------------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_few_shot", type=int, default=None)
    return parser.parse_args()


# --------------------------------------------------
# Seed
# --------------------------------------------------
def set_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

# --------------------------------------------------
# Few-shot split
# --------------------------------------------------
def split_few_shot_val(dataset, num_few_shot, grid_size):

    total_tokens = grid_size * grid_size
    min_tokens = int(total_tokens * 0.01)

    eligible = []

    for i, batch in enumerate(dataset):
        label = batch[1]
        label = (label > 0).float().unsqueeze(0)
        label = F.interpolate(label, (grid_size, grid_size), mode="nearest")
        if label.sum() >= min_tokens:
            eligible.append(i)

    if len(eligible) < num_few_shot:
        raise ValueError("Not enough eligible few-shot samples.")

    selected = random.sample(eligible, num_few_shot)
    remaining = [i for i in range(len(dataset)) if i not in selected]

    return Subset(dataset, selected), Subset(dataset, remaining)


# --------------------------------------------------
# Token collection
# --------------------------------------------------
def collect_semantic_tokens(
    dataloader,
    embedding_model,
    device,
    use_object_mask=True,
    bg_ratio=0.25,
):
    X, y = [], []
    embedding_model.eval()

    with torch.no_grad():
        for batch in dataloader:

            if len(batch) == 4: # AdaCLIP returns cls_name for prompting
                image, labels, _, cls_name = batch
                semantic = embedding_model.get_semantic_tokens(
                    image.to(device), cls_name
                )
                if use_object_mask:
                    object_mask = embedding_model.get_object_mask(image, cls_name)
                    #import cv2
                    #cv2.imwrite(f"object_mask_{cls_name[0]}.png", (object_mask[0].cpu().numpy().reshape(37, 37)*255).astype(np.uint8))
                else:
                    object_mask = None

            else:
                image, labels, _ = batch
                semantic = embedding_model.get_semantic_tokens(
                    image.to(device)
                )
                if use_object_mask:
                    object_mask = embedding_model.get_object_mask(image)
                else:
                    object_mask = None

            labels = labels.to(device)
            N, HW, _ = semantic.shape
            grid_size = int(HW ** 0.5)

            # resize labels
            labels = (labels > 0).float()
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            labels = F.interpolate(
                labels, (grid_size, grid_size), mode="nearest"
            ).squeeze(1)
            labels = labels.reshape(labels.shape[0], -1)
            # labels: [N, HW]

            for i in range(N):
                sem_i = semantic[i]
                lab_i = labels[i]

                if not use_object_mask:
                    X.append(sem_i.cpu())
                    y.append(lab_i.cpu())
                    continue

                obj_i = object_mask[i]

                abnormal_idx = torch.where(lab_i > 0)[0]
                normal_fg_idx = torch.where((lab_i == 0) & obj_i)[0]
                normal_bg_idx = torch.where((lab_i == 0) & (~obj_i))[0]

                idx_list = []
                if len(abnormal_idx) > 0:
                    idx_list.append(abnormal_idx)

                num_abnormal = len(abnormal_idx)
                remaining_budget = lab_i.shape[0] - num_abnormal

                if remaining_budget > 0:
                    bg_budget = int(remaining_budget * bg_ratio)
                    fg_budget = remaining_budget - bg_budget

                    sampled = []

                    if len(normal_fg_idx) > 0 and fg_budget > 0:
                        perm = torch.randperm(len(normal_fg_idx), device=lab_i.device)
                        sampled.append(normal_fg_idx[perm[:fg_budget]])

                    if len(normal_bg_idx) > 0 and bg_budget > 0:
                        perm = torch.randperm(len(normal_bg_idx), device=lab_i.device)
                        sampled.append(normal_bg_idx[perm[:bg_budget]])

                    if len(sampled) > 0:
                        idx_list.extend(sampled)

                if len(idx_list) == 0:
                    continue

                idx = torch.cat(idx_list)
                
                X.append(sem_i[idx].cpu())
                y.append(lab_i[idx].cpu())

    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def semantic_score_from_tokens(X_2L, num_layers, mode):
    """
    X_2L: [N, 2L] semantic logits
    Returns: [N] anomaly probability
    """
    if mode == "anomalyclip":
        return layerwise_softmax_mean(X_2L, num_layers)

    elif mode == "adaclip":
        return aggregated_softmax(X_2L, num_layers)

    else:
        raise ValueError(f"Unknown semantic_score_mode: {mode}")


def layerwise_softmax_mean(X_2L, num_layers, temperature=1.0):
    """
    X_2L: Tensor [N, 2L] = concatenated (s_normal, s_abnormal) per layer
    Returns: numpy array [N] anomaly score
    """
    X_layers = X_2L.view(-1, num_layers, 2)          # [N, L, 2]
    X_layers = X_layers / temperature
    probs = torch.softmax(X_layers, dim=-1)[:, :, 1]  # [N, L]
    return probs.mean(dim=1).cpu().numpy()            # [N]


def aggregated_softmax(X_2L, num_layers):
    """
    Aggregate logits across layers BEFORE softmax
    """
    X_layers = X_2L.view(-1, num_layers, 2)   # [N, L, 2]
    logits = X_layers.mean(dim=1)             # [N, 2]
    probs = torch.softmax(logits, dim=-1)[:, 1]
    return probs.cpu().numpy()



# --------------------------------------------------
# Softmax baseline calibration
# --------------------------------------------------
def softmax_calib(scores, labels):

    labels = labels.numpy()
    best_thr, best_dice = 0.5, -1

    for thr in np.linspace(0, 1, 101):
        pred = (scores >= thr).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        if dice > best_dice:
            best_dice = dice
            best_thr = thr

    return best_thr


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    args = parse_args()
    cfg = load_config(args.config)

    if args.num_few_shot is not None:
        cfg.num_few_shot = args.num_few_shot

    set_seed(cfg.seed)
    device = setup_device(cfg.device)

    # -------------------------
    # Load embedding model
    # -------------------------
    if cfg.score_model.name == "anomalyclip":
        ## AnomalyCLIP
        AnomalyCLIP_parameters = {
            "Prompt_length": cfg.score_model.anomalyclip.clip_n_ctx, 
            "learnabel_text_embedding_depth": cfg.score_model.anomalyclip.clip_depth, 
            "learnabel_text_embedding_length": cfg.score_model.anomalyclip.clip_t_n_ctx
        }  
        embedding_model = AnomalyCLIP_Inference(
            checkpoint_path=cfg.score_model.anomalyclip.checkpoint,
            patch_size=cfg.embedding.patch_size,
            device=device,
            design_details=AnomalyCLIP_parameters
        ).to(device)

    elif cfg.score_model.name == "adaclip":
        # Prepare model
        config_path = os.path.join('./external/AdaCLIP/model_configs', f'ViT-L-14-336.json')
        with open(config_path, 'r') as f:
            model_configs = json.load(f)
        # Set up the feature hierarchy
        n_layers = model_configs['vision_cfg']['layers']
        substage = n_layers // 4
        features_list = [substage, substage * 2, substage * 3, substage * 4]
        embedding_model = AdaCLIP_Inference(
            backbone=cfg.score_model.adaclip.backbone,
            feat_list=features_list,
            input_dim=model_configs['vision_cfg']['width'],
            output_dim=model_configs['embed_dim'],
            device=device,
            image_size=cfg.embedding.resolution,
            patch_size = cfg.embedding.patch_size,
            prompting_depth=cfg.score_model.adaclip.prompting.depth,
            prompting_length=cfg.score_model.adaclip.prompting.length,
            prompting_branch=cfg.score_model.adaclip.prompting.branch,
            prompting_type=cfg.score_model.adaclip.prompting.type,
            use_hsf=cfg.score_model.adaclip.use_hsf,
            k_clusters=cfg.score_model.adaclip.k_clusters,
        ).to(device)
        ckt_path = cfg.score_model.adaclip.checkpoint
        embedding_model.load_weights(ckt_path)
        print("loading checkpoint...done...")
    else:
        raise ValueError("Unknown semantic_score_mode")

    grid_size = cfg.embedding.resolution // cfg.embedding.patch_size

    # -------------------------
    # Loop objects
    # -------------------------
    for object_name in cfg.dataset.objects:

        experiment_dir = makedirs(cfg.logging.base_dir, object_name)
        results = {}

        for repeat_id in range(cfg.num_fewshot_repeats):

            set_seed(cfg.seed + repeat_id)

            # ---- datasets ----
            val_normal, test_normal = get_datasets(
                cfg, object_name, ["good"], ["val", "test"]
            )

            val_anom, test_anom = get_datasets(
                cfg,
                object_name,
                cfg.dataset.object_anomalies[object_name],
                ["val", "test"],
            )

            fewshot_set, val_rest = split_few_shot_val(
                val_anom, cfg.num_few_shot, grid_size
            )

            loader_fs = DataLoader(
                fewshot_set,
                batch_size=cfg.embedding.batch_size,
                shuffle=True,
            )

            # ---- collect tokens ----
            X_train, y_train = collect_semantic_tokens(
                loader_fs, embedding_model, device
            )

            # ---- train classifier ----
            classifier = train_few_shot_classifier(
                X_train,
                y_train,
                device=device,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )

            # ---- baseline threshold ----
            X_tokens_fs, y_tokens_fs = collect_semantic_tokens(
                loader_fs, 
                embedding_model,
                device,
                use_object_mask=False, # use all available tokens for calibration
            )            
            with torch.no_grad():
                num_layers = X_tokens_fs.shape[1] // 2
                softmax_train_scores = semantic_score_from_tokens(X_tokens_fs, num_layers, mode=cfg.score_model.name)  

            best_thr = softmax_calib(softmax_train_scores, y_tokens_fs)

            # ---- evaluation ----
            if cfg.score_model.name == "anomalyclip":
                stats_df = test_fewshot_model(
                    embedding_model,
                    classifier,
                    best_thr,
                    DataLoader(test_normal),
                    DataLoader(test_anom),
                    cfg,
                    device,
                    map_key_baseline="anomalyclip_map",
                    evaluate_fn=evaluate_fewshot,
                )
            else:
                stats_df = test_fewshot_model(
                    embedding_model,
                    classifier,
                    best_thr,
                    DataLoader(test_normal),
                    DataLoader(test_anom),
                    cfg,    
                    device,
                    map_key_baseline="adaclip_map",
                    evaluate_fn=evaluate_fewshot_adaclip,
                )

            # ---- aggregate ----
            if not results:
                for k in stats_df.index:
                    results[k] = {m: [] for m in stats_df.columns}

            for k in stats_df.index:
                for m in stats_df.columns:
                    results[k][m].append(stats_df.loc[k, m])

        # ---- compute mean/std ----
        rows = []
        for k, metrics in results.items():
            row = {"map_key": k}
            for m, values in metrics.items():
                values = np.array(values, dtype=float)
                row[f"{m}_mean"] = np.nanmean(values)
                row[f"{m}_std"] = np.nanstd(values)
            rows.append(row)

        final_df = pd.DataFrame(rows)
        final_df.to_csv(
            os.path.join(
                experiment_dir,
                f"stats_test_{cfg.num_few_shot}_runs.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    main()
