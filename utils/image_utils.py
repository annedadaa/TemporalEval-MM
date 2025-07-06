import os
import torch
import numpy as np
from typing import List, Union, Tuple
from PIL import Image
import matplotlib.pyplot as plt

def unnormalize(t: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize a tensor image normalized with mean=0.5 and std=0.5 per channel.
    """
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(t.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(t.device)
    return t * std + mean

def to_numpy(img: Union[torch.Tensor, Image.Image]) -> np.ndarray:
    """
    Convert an image from a torch.Tensor or PIL.Image to a numpy ndarray suitable for saving or visualization.
    """
    if isinstance(img, torch.Tensor):
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = unnormalize(img).permute(1, 2, 0).detach().cpu().numpy()
        elif img.dim() == 2:
            img = img.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {img.shape}")
        return (img * 255).clip(0, 255).astype(np.uint8)
    elif isinstance(img, Image.Image):
        return np.array(img)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

def visualize_stacked_frames(frames: List[Union[torch.Tensor, List[torch.Tensor]]], output_path: str, filename: str) -> None:
    """
    Save a horizontally stacked image composed of a list of frames.
    """
    all_imgs = []
    for frame in frames:
        imgs = frame if isinstance(frame, list) else [frame]
        all_imgs.extend(imgs)

    np_imgs = [to_numpy(img) for img in all_imgs]
    full_image = np.hstack(np_imgs)

    os.makedirs(output_path, exist_ok=True)
    Image.fromarray(full_image).save(os.path.join(output_path, filename))

def uniform_sample(l: List, n: int) -> List:
    """
    Uniformly samples n elements from the list l by dividing the list into equal segments
    and picking the middle element from each segment.
    """
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def approximate_smart_resize(
    height: int,
    width: int,
    image_factor: int,
    min_pixels: int,
    max_pixels: int
) -> Tuple[int, int]:
    """
    Compute new width and height to resize an image approximately while preserving the aspect ratio,
    limiting the total number of pixels between min_pixels and max_pixels divided by image_factor.
    """
    target_pixels = max(min_pixels, min(max_pixels, height * width // image_factor))
    aspect_ratio = width / height
    new_height = int((target_pixels / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    return new_width, new_height
