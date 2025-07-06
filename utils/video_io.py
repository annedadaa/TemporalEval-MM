import os
import json
import torch

from typing import Union, List, Dict

from models.llava import LLaVAOneVisionModel
from models.qwen import Qwen2VLModel
from utils.config import Config


def encode_concat_video_llava(
    config: Config, model_vqa: Union[LLaVAOneVisionModel, Qwen2VLModel], video1_path: str, video2_path: str
    ) -> torch.Tensor:
    """
    Concatenate frames from two videos and return as a single tensor.
    """
    video1 = os.path.join(config.general.video_dir, video1_path)
    video2 = os.path.join(config.general.video_dir, video2_path)

    frames1 = model_vqa.load_images([video1], config.experiment.max_frames_num)
    frames2 = model_vqa.load_images([video2], config.experiment.max_frames_num)
    return torch.cat([frames1[0], frames2[0]], dim=0)

def encode_single_video_llava(
    config: Config, model_vqa: Union[LLaVAOneVisionModel, Qwen2VLModel], video_path: str
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Load a single video and return its frames as a tensor.
    """
    path = os.path.join(config.general.video_dir, video_path)
    return model_vqa.load_images([path], config.experiment.max_frames_num)[0]

def encode_concat_video_qwen(
    config: Config, model_vqa: Union[LLaVAOneVisionModel, Qwen2VLModel], video1_path: str, video2_path: str
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Concatenate frames from two videos and return as a list of tensors.
    """
    video1 = os.path.join(config.general.video_dir, video1_path)
    video2 = os.path.join(config.general.video_dir, video2_path)
    frames1, frames2 = model_vqa.load_images([video1, video2], config.experiment.max_frames_num)
    return frames1 + frames2

def encode_single_video_qwen(
    config: Config, model_vqa: Union[LLaVAOneVisionModel, Qwen2VLModel], video_path: str
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Load a single video and return its frames as a list of tensors.
    """
    path = os.path.join(config.general.video_dir, video_path)
    return model_vqa.load_images([path], config.experiment.max_frames_num)[0]

def read_video_pairs(pairs_jsonl) -> List[Dict[str, str]]:
    """
    Read video pairs from a JSONL file and return them as a list of dictionaries.
    """
    with open(pairs_jsonl, "r") as f:
        return [{"video1": p["video1"], "video2": p["video2"]} for p in map(json.loads, f)]
