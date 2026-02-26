import os
import pandas as pd
from .metrics import compute_metrics
from .evaluate_clip import evaluate_fewshot, evaluate_fewshot_adaclip


def test_fewshot_model(
    embedding_model,
    classifier,
    calibrated_threshold,
    dataloader_test_normal,
    dataloader_test_anomalous,
    cfg,
    device,
    evaluate_fn,
    map_key_baseline,
):
    """
    Evaluation:
    - run baseline + few-shot
    - compute metrics
    - return DataFrame
    """

    # Inference
    stats_normal = evaluate_fn(
        dataloader_test_normal,
        embedding_model,
        classifier,
        device,
        cfg,
    )

    stats_anomalous = evaluate_fn(
        dataloader_test_anomalous,
        embedding_model,
        classifier,
        device,
        cfg,
    )

    score_map_keys = [
        map_key_baseline,
        "fewshot_semantic_map",
    ]
    stats_df = pd.DataFrame(
        index=score_map_keys,
        columns=[
            "auroc",
            "recall",
            "precision",
            "f1",
            "balacc",
            "pro",
            "aupro",
        ],
    )

    # ---------------------------------------------
    # Compute metrics per map
    # ---------------------------------------------
    for map_key in score_map_keys:

        thr = (
            calibrated_threshold
            if map_key == map_key_baseline
            else 0.5
        )

        # Compute metrics
        stats_df = compute_metrics(
            stats_df,
            map_key, 
            stats_normal,
            stats_anomalous,
            thr,
        )

    return stats_df

