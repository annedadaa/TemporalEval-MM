import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from utils.config_constants import QWEN2_VL_MODELS
from utils.image_utils import approximate_smart_resize, uniform_sample


class Qwen2VLModel:
    def __init__(self, model_name="Qwen2.5-VL-7B-Instruct", device="cuda:0", cache_dir=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN2_VL_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_path = self.model_info["model"]["path"]

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.model_info["model"]["torch_dtype"],
            attn_implementation=self.model_info["model"]["attn_implementation"],
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_info["tokenizer"]["path"]
        )

        self.model.eval()

    def load_images(self, paths: List[str], max_frames_num: int, fps_factor: int = 1) \
            -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        target_size = None

        for path in paths:
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                frames = self.load_video(path, max_frames_num, fps_factor, target_size=target_size)

                if target_size is None and len(frames) > 0:
                    target_size = frames[0].size

                processed_data.append(frames)
            elif path.lower().endswith(".npy"):
                np_array = np.load(path)
                if np_array.ndim == 3:
                    image = Image.fromarray(np_array.astype("uint8"), "RGB")
                    processed_data.append([image])
                elif np_array.ndim == 4:
                    frames = [Image.fromarray(f.astype("uint8"), "RGB") for f in np_array]
                    processed_data.append(frames)
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:
                image = Image.open(path).convert("RGB")
                processed_data.append([image])

        return processed_data

    def load_video(self, video_path, max_frames_num, fps_factor=1, target_size=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = max(1, round(vr.get_avg_fps() / fps_factor))
        frame_idx = list(range(0, len(vr), sample_fps))

        if len(frame_idx) >= max_frames_num:
            frame_idx = uniform_sample(frame_idx, max_frames_num)
        else:
            last_idx = frame_idx[-1] if frame_idx else len(vr) - 1
            frame_idx += [last_idx] * (max_frames_num - len(frame_idx))

        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(f.astype("uint8")) for f in frames]

        orig_width, orig_height = frames[0].size

        # Smart resizing logic
        IMAGE_FACTOR = 28
        VIDEO_MIN_PIXELS = 128 * 28 * 28
        VIDEO_MAX_PIXELS = 768 * 28 * 28
        FRAME_FACTOR = 2
        VIDEO_TOTAL_PIXELS = 640 * 360 * 28 * 28

        nframes = len(frames)
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR),
            int(VIDEO_MIN_PIXELS * 1.05)
        )

        if target_size is None:
        # Compute target size as before
            target_width, target_height = approximate_smart_resize(
                orig_height, orig_width,
                image_factor=IMAGE_FACTOR,
                min_pixels=VIDEO_MIN_PIXELS,
                max_pixels=max_pixels,
            )
        else:
            target_width, target_height = target_size

        resized_frames = [
            frame.resize((target_width, target_height), Image.BICUBIC)
            for frame in frames
        ]

        return resized_frames

    @torch.no_grad()
    def generate(
        self,
        data: List[str] = None,
        max_new_tokens: int = 256,
        conditioned_system_prompt: str = None,
    ) -> List[str]:
        generated_texts = []

        inputs_list = data if data else [None]

        for image in inputs_list:
            text = "Output a similarity score from 1 to 5."

            if image is None:
                messages = [
                    {"role": "system", "content": conditioned_system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": text}]},
                ]
                image_inputs, video_inputs = None, None
            else:
                messages = [
                    {"role": "system", "content": conditioned_system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": text}, image]},
                ]

                image_inputs, video_inputs = process_vision_info(messages, return_video_kwargs=False)

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
        )

            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                generated_texts.append(text)

            torch.cuda.empty_cache()

        return generated_texts
