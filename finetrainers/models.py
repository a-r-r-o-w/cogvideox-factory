from typing import Any, Dict

from .hunyuan_video import HUNYUAN_VIDEO_T2V_LORA_CONFIG
from .ltx_video import LTX_VIDEO_T2V_LORA_CONFIG


SUPPORTED_MODEL_CONFIGS = {
    "hunyuan_video": {
        "lora": HUNYUAN_VIDEO_T2V_LORA_CONFIG,
    },
    "ltx_video": {
        "lora": LTX_VIDEO_T2V_LORA_CONFIG,
    },
}


def get_config_from_model_name(model_name: str, training_type: str) -> Dict[str, Any]:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if training_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Training type {training_type} not supported for model {model_name}. Supported training types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][training_type]
