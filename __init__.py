NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Core nodes - each import is isolated so one failure doesn't break the rest
try:
    from .any_text_generate import AnyTextGenerate
    NODE_CLASS_MAPPINGS["AnyTextGenerate"] = AnyTextGenerate
    NODE_DISPLAY_NAME_MAPPINGS["AnyTextGenerate"] = "Any Text Generate (with System Prompt)"
except Exception as e:
    print(f"[anyMODE] Failed to load AnyTextGenerate: {e}")

try:
    from .comfyui_lora_blend import LoraBlend
    NODE_CLASS_MAPPINGS["LoraBlender"] = LoraBlend
    NODE_DISPLAY_NAME_MAPPINGS["LoraBlender"] = "Lora Blender"
except Exception as e:
    print(f"[anyMODE] Failed to load LoraBlend: {e}")

try:
    from .lora_xy_grid import LoraXYIntegratedSampler, LoraXYIntegratedSamplerCustom
    NODE_CLASS_MAPPINGS["LoraXYIntegratedSampler"] = LoraXYIntegratedSampler
    NODE_CLASS_MAPPINGS["LoraXYIntegratedSamplerCustom"] = LoraXYIntegratedSamplerCustom
    NODE_DISPLAY_NAME_MAPPINGS["LoraXYIntegratedSampler"] = "LoRA XY Integrated Sampler"
    NODE_DISPLAY_NAME_MAPPINGS["LoraXYIntegratedSamplerCustom"] = "LoRA XY Integrated Sampler (Custom)"
except Exception as e:
    print(f"[anyMODE] Failed to load LoraXY nodes: {e}")

try:
    from .quantizer_node import AnyModeQuantizer
    NODE_CLASS_MAPPINGS["AnyModeQuantizer"] = AnyModeQuantizer
    NODE_DISPLAY_NAME_MAPPINGS["AnyModeQuantizer"] = "anyMODE Quantizer"
except Exception as e:
    print(f"[anyMODE] Failed to load AnyModeQuantizer: {e}")

try:
    from .calibration_nodes import AnyModeCalibration, AnyModeSaveQuantizedWithCalibration
    NODE_CLASS_MAPPINGS["AnyModeCalibration"] = AnyModeCalibration
    NODE_CLASS_MAPPINGS["AnyModeSaveQuantizedWithCalibration"] = AnyModeSaveQuantizedWithCalibration
    NODE_DISPLAY_NAME_MAPPINGS["AnyModeCalibration"] = "anyMODE Calibration (Toggle)"
    NODE_DISPLAY_NAME_MAPPINGS["AnyModeSaveQuantizedWithCalibration"] = "anyMODE Save Calibrated Model"
except Exception as e:
    print(f"[anyMODE] Failed to load calibration nodes: {e}")

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
