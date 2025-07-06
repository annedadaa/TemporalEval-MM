import copy
from typing import List, Optional, Union

import numpy as np
import torch
from torch.nn import DataParallel
from PIL import Image
from decord import VideoReader, cpu

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from utils.config_constants import LLAVA_OV_MODELS


class LLaVAOneVisionModel:
    def __init__(
        self,
        model_name: str = "LLaVA-OneVision-Qwen2-7B",
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = LLAVA_OV_MODELS[model_name]
        self.conversational_style = self.model_info["model"]["conversation"]
        self.load_model()

    def load_model(self) -> None:
        """
        Load the pretrained model and associated tokenizer and processor,
        move the model to the target device and set it to evaluation mode.
        """
        model_path = self.model_info["model"]["path"]

        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path,
            None,
            "llava_qwen",
            device_map="auto",
            attn_implementation="sdpa",
        )

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No GPU available")

        self.model.eval()

    def load_images(
        self, paths: List[str], num_frames: int = 16
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Load images or videos from given paths and preprocess them.
        """
        processed_data = []
        for path in paths:
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file
                video_frames = self.load_video(path, num_frames)
                frames = (
                    self.processor.preprocess(video_frames, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .to(self.device)
                )
                processed_data.append(frames)
            elif path.lower().endswith(".npy"):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype("uint8"), "RGB")
                    image_tensor = process_images(
                        [image], self.processor, self.model.config
                    )
                    image_tensor = [
                        _image.to(dtype=torch.float16, device=self.device)
                        for _image in image_tensor
                    ]
                    processed_data.append(image_tensor[0])
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [
                        Image.fromarray(frame.astype("uint8"), "RGB")
                        for frame in np_array
                    ]
                    frames_tensor = (
                        self.processor.preprocess(frames, return_tensors="pt")[
                            "pixel_values"
                        ]
                        .half()
                        .to(self.device)
                    )
                    processed_data.append(frames_tensor)
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert("RGB")
                image_tensor = process_images(
                    [image], self.processor, self.model.config
                )
                image_tensor = [
                    _image.to(dtype=torch.float16, device=self.device)
                    for _image in image_tensor
                ]
                processed_data.append(image_tensor[0])

        return processed_data

    def load_video(self, video_path: str, max_frames_num: int) -> List[Image.Image]:
        """
        Load uniformly sampled frames from a video file.
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(frame) for frame in spare_frames]

    def format_question(self, question: str, has_media: bool = False) -> str:
        """
        Format the question string within the conversation template.
        """
        conv = copy.deepcopy(conv_templates[self.conversational_style])
        prefix = (DEFAULT_IMAGE_TOKEN + "\n") if has_media else ""
        conv.append_message(conv.roles[0], prefix + question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate(
        self,
        data: Union[torch.Tensor, List[str]] = None,
        max_new_tokens: int = 256,
        conditioned_system_prompt: str = None,
    ) -> List[str]:
        """
        Generate response text from the model based on input data and prompt.
        """
        has_media = data is not None
        prompt = self.format_question(conditioned_system_prompt, has_media=has_media)

        generated_texts = []

        if has_media:
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)
        else:
            input_ids = self.tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(self.device)

        # Prepare generation arguments
        generate_args = dict(
            inputs=input_ids,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )

        if has_media:
            if isinstance(data, torch.Tensor) and data.dim() == 4:  # Video
                image_sizes = [data.shape[2:] for _ in range(data.shape[0])]
                modalities = ["video"]
            else:  # Image
                image_sizes = [data.shape[1:]]
                modalities = None

            generate_args.update(dict(
            images=[data],
            image_sizes=image_sizes,
            modalities=modalities,
            ))

        with torch.no_grad():
            outputs = self.model.generate(**generate_args)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(text.strip())

        torch.cuda.empty_cache()

        return generated_texts
