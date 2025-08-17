import os
import json
import torch

from typing import Union, List, Dict

from models.llava import LLaVAOneVisionModel
from models.qwen import Qwen2VLModel
from models.internvl import InternVL3Model
from utils.config import Config


def encode_concat_video_llava(
    config: Config, model_vqa: LLaVAOneVisionModel, video1_path: str, video2_path: str
    ) -> torch.Tensor:
    """
    Concatenate frames from two videos and return as a single tensor for the LlaVa model.
    """
    video1 = os.path.join(config.general.video_dir, video1_path)
    video2 = os.path.join(config.general.video_dir, video2_path)

    frames1 = model_vqa.load_images([video1], config.experiment.max_frames_num, config.experiment.shuffle_frames)
    frames2 = model_vqa.load_images([video2], config.experiment.max_frames_num, config.experiment.shuffle_frames)
    return torch.cat([frames1[0], frames2[0]], dim=0)

def encode_single_video_llava(
    config: Config, model_vqa: LLaVAOneVisionModel, video_path: str
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Load a single video and return its frames as a tensor for the LlaVa model.
    """
    path = os.path.join(config.general.video_dir, video_path)
    return model_vqa.load_images([path], config.experiment.max_frames_num, config.experiment.shuffle_frames)[0]

def encode_concat_video_qwen(
    config: Config, model_vqa: Qwen2VLModel, video1_path: str, video2_path: str
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Concatenate frames from two videos and return as a list of tensors for the Qwen model.
    """
    video1 = os.path.join(config.general.video_dir, video1_path)
    video2 = os.path.join(config.general.video_dir, video2_path)
    frames1, frames2 = model_vqa.load_images([video1, video2], config.experiment.max_frames_num, config.experiment.shuffle_frames)
    return frames1 + frames2

def encode_single_video_qwen(
    config: Config, model_vqa: Qwen2VLModel, video_path: str
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Load a single video and return its frames as a list of tensors for the Qwen model.
    """
    path = os.path.join(config.general.video_dir, video_path)
    return model_vqa.load_images([path], config.experiment.max_frames_num, config.experiment.shuffle_frames)[0]

def encode_concat_video_internvl(
    config: Config, model_vqa: InternVL3Model, video1_path: str, video2_path :str) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Concatenate frames from two videos and return as a list of tensors for the InternVL model.
    """

    video1_path = os.path.join(config.general.video_dir, video1_path)
    video2_path = os.path.join(config.general.video_dir, video2_path)

    video_frames1 = model_vqa.encode_video(video1_path, config.experiment.max_frames_num, config.experiment.shuffle_frames)
    video_frames2 = model_vqa.encode_video(video2_path, config.experiment.max_frames_num, config.experiment.shuffle_frames)
            
    video_frame1, video_num_patches1 = video_frames1
    video_frame2, video_num_patches2 = video_frames2

    video_frames = torch.cat([video_frame1, video_frame2], dim=0)
    video_num_patches = video_num_patches1 + video_num_patches2
    return [video_frames, video_num_patches]

def encode_single_video_internvl(
    config: Config, model_vqa: InternVL3Model, video_path: str) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Load a single video and return its frames as a list of tensors for the InternVL model.
    """

    path = os.path.join(config.general.video_dir, video_path)
    video_frames, video_num_patches = model_vqa.encode_video(path, config.experiment.max_frames_num, config.experiment.shuffle_frames)
    return [video_frames, video_num_patches]

def read_video_pairs(pairs_jsonl) -> List[Dict[str, str]]:
    """
    Read video pairs from a JSONL file and return them as a list of dictionaries.
    """
    with open(pairs_jsonl, "r") as f:
        return [{"video1": p["video1"], "video2": p["video2"]} for p in map(json.loads, f)]
