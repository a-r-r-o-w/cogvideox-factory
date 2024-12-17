from typing import Any, Dict

from .ltx_video import LTX_VIDEO_T2V_CONFIG


SUPPORTED_MODEL_CONFIGS = {
    "ltx_video": LTX_VIDEO_T2V_CONFIG,
}


def get_config_from_model_name(model_name: str) -> Dict[str, Any]:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name]
