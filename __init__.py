from .any_text_generate import AnyTextGenerate
from .comfyui_lora_blend import LoraBlend
from .lora_xy_grid import LoraXYIntegratedSampler, LoraXYIntegratedSamplerCustom
from .quantizer_node import AnyModeQuantizer
from .calibration_nodes import AnyModeCalibration, AnyModeSaveQuantizedWithCalibration



WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "AnyTextGenerate": AnyTextGenerate,
    "LoraBlender": LoraBlend,
    "LoraXYIntegratedSampler": LoraXYIntegratedSampler,
    "LoraXYIntegratedSamplerCustom": LoraXYIntegratedSamplerCustom,
    "AnyModeQuantizer": AnyModeQuantizer,
    "AnyModeCalibration": AnyModeCalibration,
    "AnyModeSaveQuantizedWithCalibration": AnyModeSaveQuantizedWithCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyTextGenerate": "Any Text Generate (with System Prompt)",
    "LoraBlender": "Lora Blender",
    "LoraXYIntegratedSampler": "LoRA XY Integrated Sampler",
    "LoraXYIntegratedSamplerCustom": "LoRA XY Integrated Sampler (Custom)",
    "AnyModeQuantizer": "anyMODE Quantizer",
    "AnyModeCalibration": "anyMODE Calibration (Toggle)",
    "AnyModeSaveQuantizedWithCalibration": "anyMODE Save Calibrated Model",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
