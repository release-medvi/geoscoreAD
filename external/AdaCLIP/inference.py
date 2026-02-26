
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .adaclip import *
from .custom_clip import create_model_and_transforms


class AdaCLIP_Inference(nn.Module):
    def __init__(
        self,
        backbone,
        feat_list,
        input_dim,
        output_dim,
        device,
        image_size,
        patch_size,
        prompting_depth=3,
        prompting_length=2,
        prompting_branch='VL',
        prompting_type='SD',
        use_hsf=True,
        k_clusters=20,
    ):
        super().__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.patch_size = patch_size

        # --- Load CLIP backbone exactly as original ---
        freeze_clip, _, self.preprocess = create_model_and_transforms(
            backbone,
            image_size,
            pretrained='openai'
        )

        freeze_clip = freeze_clip.to(device)
        freeze_clip.eval()

        # --- Build AdaCLIP model exactly as original ---
        self.clip_model = AdaCLIP(
            freeze_clip=freeze_clip,
            text_channel=output_dim,
            visual_channel=input_dim,
            prompting_length=prompting_length,
            prompting_depth=prompting_depth,
            prompting_branch=prompting_branch,
            prompting_type=prompting_type,
            use_hsf=use_hsf,
            k_clusters=k_clusters,
            output_layers=feat_list,
            device=device,
            image_size=image_size
        ).to(device)

        self.clip_model.eval()

        # --- Keep identical preprocessing modification ---
        self.preprocess.transforms[0] = transforms.Resize(
            size=(image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None
        )

        self.preprocess.transforms[1] = transforms.CenterCrop(
            size=(image_size, image_size)
        )

    def load_weights(self, path):
        """
        Load pretrained AdaCLIP weights (same behavior as original load)
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)

    def get_semantic_tokens(self, image, cls_name):
        image = image.to(next(self.clip_model.parameters()).device)

        grid_size = image.shape[-1] // self.patch_size

        anomaly_maps, _ = self.clip_model(
            image, cls_name, aggregation=False
        )

        semantic_layers = []
        for amap in anomaly_maps:
            amap = F.adaptive_avg_pool2d(amap, (grid_size, grid_size))
            B, _, H, W = amap.shape
            amap = amap.permute(0, 2, 3, 1).reshape(B, H * W, 2)
            semantic_layers.append(amap)

        return torch.cat(semantic_layers, dim=-1)


    def get_object_mask(self, image, cls_name):
        device = next(self.clip_model.parameters()).device
        image = image.to(device)

        grid_size = image.shape[-1] // self.patch_size

        _, patch_tokens, _ = self.clip_model.extract_feat(image, cls_name)

        return self.clip_model.compute_background_mask_adaclip(
            patch_tokens=patch_tokens,
            grid_size=(grid_size, grid_size),
        )


    @torch.no_grad()
    def forward(self, image, cls_name, aggregation=True):
        """
        Zero-shot inference.

        Returns:
            anomaly_map
            anomaly_score
        """
        self.clip_model.eval()
        return self.clip_model(image, cls_name, aggregation=aggregation)
