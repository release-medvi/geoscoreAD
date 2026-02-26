import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from sklearn.decomposition import PCA

from .model_load import load
from .prompt_ensemble import AnomalyCLIP_PromptLearner

class AnomalyCLIP_Inference(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        patch_size, 
        device,
        design_details,
        model_name="ViT-L/14@336px",
    ):
        super().__init__()

        self.device = device

        # -------------------------------------------------
        # Load base CLIP backbone
        # -------------------------------------------------
        self.embedding_model, _ = load(
            model_name,
            device=device,
            design_details=design_details
        )
        self.patch_size = patch_size
        # -------------------------------------------------
        # Load Prompt Learner
        # -------------------------------------------------
        self.prompt_learner = AnomalyCLIP_PromptLearner(
            self.embedding_model.to("cpu"),
            design_details
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])

        self.prompt_learner.to(device)
        self.embedding_model.to(device)

        # Replace DAPM layer (as original code)
        self.embedding_model.visual.DAPM_replace(DPAM_layer=20)

        # -------------------------------------------------
        # Precompute universal text features
        # -------------------------------------------------
        self.text_features = self._build_text_features()

    # =====================================================
    # Build universal text features once
    # =====================================================
    @torch.no_grad()
    def _build_text_features(self):

        prompts, tokenized_prompts, compound_prompts_text = \
            self.prompt_learner(cls_id=None)

        text_features = self.embedding_model.encode_text_learn(
            prompts,
            tokenized_prompts,
            compound_prompts_text
        ).float()

        # reshape to (2, D)
        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2),
            dim=1
        )

        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.squeeze(0)  # (2, D)

        return text_features.to(self.device)

    # =====================================================
    # Extract multi-layer visual features
    # =====================================================
    def extract_features(self, image):
        return self.embedding_model.extract_features(image)


    def crop_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Cropping the image to a size that is a multiple of the patch size.

        :param torch.Tensor image: Tensor with shape NCHW.
        :return torch.Tensor: Image cropped to multiple of patch size.
        :return tuple: Image size in number of patches.
        """
        _, _, height, width = image.shape
        cropped_height = height - height % self.patch_size
        cropped_width = width - width % self.patch_size

        image = image[:, :, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        return image, grid_size

    ## added for DPMM comparison
    def compute_background_mask(
        self,
        features: torch.Tensor,   # [N, HW, C]
        grid_size: tuple,         # (H, W)
        masking: bool = False,
        kernel_size: int = 3,
    ) -> torch.Tensor:

        N, S, D = features.shape
        H, W = grid_size
        device = features.device

        if not masking:
            return torch.ones((N, S), dtype=torch.bool, device=device)

        mask = np.zeros((N, H, W), dtype=np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for i in range(N):
            feats_i = features[i].cpu().numpy()  # [HW, C]

            # --- z-score normalization ---
            feats_i = (feats_i - feats_i.mean(axis=0, keepdims=True)) / (
                feats_i.std(axis=0, keepdims=True) + 1e-6
            )

            # --- PCA ---
            pc = PCA(n_components=1).fit_transform(feats_i)  # [HW, 1]
            abs_pc = np.abs(pc[:, 0])                         # [HW]

            # --- adaptive threshold ---
            thr = np.percentile(abs_pc, 90)
            mask_i = (abs_pc > thr).reshape(H, W).astype(np.uint8)

            # --- morphology (2D only!) ---
            mask_i = cv2.dilate(mask_i, kernel)
            mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_CLOSE, kernel)

            # --- safety fallback ---
            if mask_i.sum() == 0:
                mask_i[:] = 1

            mask[i] = mask_i

        # flatten back to [N, HW]
        mask = torch.from_numpy(mask.reshape(N, S)).bool().to(device)
        return mask

    def get_semantic_tokens(self, image):
        image, grid_size = self.crop_image(image)
        return self.forward(image, grid_size, aggregation=False)

    def get_object_mask(self, image):
        image, grid_size = self.crop_image(image)
        feats_list = self.extract_features(image)

        return self.compute_background_mask(
            features=feats_list[-1],
            grid_size=grid_size,
            masking=True,
        )

    # =====================================================
    # Forward (semantic similarity computation)
    # =====================================================
    @torch.no_grad()
    def forward(
        self,
        image,
        grid_size,
        aggregation=True,
        temperature=0.07,
    ):

        feats_list = self.extract_features(image)
        text_features = self.text_features  # (2, D)

        H, W = grid_size
        semantic_layers = []

        for feats in feats_list:

            feats = F.normalize(feats, dim=-1)
            similarity = feats @ text_features.t()
            
            if aggregation:
                similarity = similarity / temperature
                similarity = similarity.softmax(dim=-1)

            semantic_layers.append(similarity)

        if not aggregation:
            return torch.cat(semantic_layers, dim=-1)  # [B, HW, 2L]

        stacked = torch.stack(semantic_layers, dim=0)
        aggregated = stacked.mean(dim=0)

        anomaly_map = aggregated[:, :, 1]  # abnormal channel

        B = anomaly_map.shape[0]
        return anomaly_map.reshape(B, H, W)  