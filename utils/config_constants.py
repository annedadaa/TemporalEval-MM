import torch

LLAVA_OV_MODELS = {
    "LLaVA-OneVision-Qwen2-7B": {
        "tokenizer": {
            "path": "lmms-lab/llava-onevision-qwen2-7b-ov"
            },
        "model": {
            "path": "lmms-lab/llava-onevision-qwen2-7b-ov",
            "conversation": "qwen_1_5",
            "image_aspect_ratio": "pad",
            },
    },
}

QWEN2_VL_MODELS = {
    "Qwen2.5-VL-7B-Instruct": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct"
            },
        "model": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            },
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INTERN_VL_MODELS = {
    "InternVL3-8b": {
        "tokenizer": {
            "path": "OpenGVLab/InternVL3-8B",
            "trust_remote_code": True,
            "use_fast": False,
        },
        "model": {
            "pretrained_model_name_or_path": "OpenGVLab/InternVL3-8B",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "use_flash_attn": True,
            "trust_remote_code": True,
            "device_map": "auto",
        },
    },
}
