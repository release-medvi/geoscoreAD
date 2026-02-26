import os
import torch
from omegaconf import OmegaConf


def makedirs(*folders):
    if len(folders) == 0:
        return "."
    
    path = os.path.join(*folders)
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path

def load_config(path):
    OmegaConf.register_new_resolver("ifequal_else", ifequal_else, replace=True)
    cfg = OmegaConf.load(path)
    return cfg


def setup_device(device):
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(device[-1])
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    return device

def ifequal_else(variable: str, expected_value: str, true_value: str, false_value: str = "", *, _root_):
    return true_value if variable == expected_value else false_value

