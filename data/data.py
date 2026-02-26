import os
from glob import glob
from natsort import natsorted
from PIL import Image

import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class Dataset(TorchDataset):
    def __init__(self, cfg, object_name, anomaly_types, split):

        self.cfg = cfg
        self.object_name = object_name

        # -------------------------
        # AdaCLIP prompt (optional)
        # -------------------------
        if cfg.score_model.name == "adaclip":
            prompt_map = {
                "bras2021": "BrainMRI",
                "liver": "LiverCT",
                "RESC": "RetinaOCT",
            }
            self.adaclip_text_prompt = prompt_map.get(cfg.dataset.name, None)
        else:
            self.adaclip_text_prompt = None

        # -------------------------
        # Collect image / label paths
        # -------------------------
        path_images_base = cfg.dataset.get(f"path_{split}_images")
        path_labels_base = cfg.dataset.get(f"path_{split}_labels")

        self.images = []
        self.labels = []

        for anomaly_type in anomaly_types:

            path_images = get_data_dir(
                path_images_base,
                cfg.dataset.data_root,
                object_name,
                anomaly_type,
            )

            path_labels = get_data_dir(
                path_labels_base,
                cfg.dataset.data_root,
                object_name,
                anomaly_type,
            )

            self.images += natsorted(glob(os.path.join(path_images, "*")))

            if path_labels:
                self.labels += natsorted(glob(os.path.join(path_labels, "*")))

        # -------------------------
        # Normalization
        # -------------------------
        if cfg.normalization.lower() == "imagenet":
            normalization = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        elif cfg.normalization.lower() == "openai":
            normalization = transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        else:
            normalization = torch.nn.Identity()

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (cfg.embedding.resolution, cfg.embedding.resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            normalization,
        ])

        self.label_transform = transforms.ToTensor() # no resizing for labels. 

    # --------------------------------------------------
    def __len__(self):
        return len(self.images)

    # --------------------------------------------------
    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert("RGB")

        if len(self.labels) > 0:
            label = Image.open(self.labels[index]).convert("L")
        else:
            label = Image.new("L", image.size)

        image = self.image_transform(image)
        label = self.label_transform(label)

        if self.adaclip_text_prompt is None:
            return image, label, self.images[index]
        else:
            return image, label, self.images[index], self.adaclip_text_prompt


# --------------------------------------------------
# Dataset factory
# --------------------------------------------------
def get_datasets(cfg, object_name, anomaly_types, splits):
    return [
        Dataset(cfg, object_name, anomaly_types, split)
        for split in splits
    ]


# --------------------------------------------------
# Path formatter
# --------------------------------------------------
def get_data_dir(path_expression, data_root, object_name, type_anomaly):
    return path_expression.format(
        data_root=data_root,
        object_name=object_name,
        type_anomaly=type_anomaly,
    )
