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
