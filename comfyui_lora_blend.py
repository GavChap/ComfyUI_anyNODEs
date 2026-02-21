# comfyui_lora_blend.py

import os
import comfy
import folder_paths

class LoraBlend:
    def __init__(self):
        self.lora1_path = None
        self.lora2_path = None
        self.blend_factor = 0.5
        self.lora1 = None
        self.lora2 = None

    @classmethod
    def INPUT_TYPES(cls):
        lora_files = folder_paths.get_filename_list("loras") # List LoRAs
        return {
            "required": {
                "lora1_file": (lora_files, {"default": lora_files[0] if lora_files else None}),  # Dropdown for LoRA 1
                "lora2_file": (lora_files, {"default": lora_files[0] if lora_files else None}),  # Dropdown for LoRA 2
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "blend_loras"
    CATEGORY = "lora"

    def blend_loras(self, model, lora1_file, lora2_file, blend_factor):
        lora1_path = folder_paths.get_full_path_or_raise("loras", lora1_file) # Construct full paths
        lora2_path = folder_paths.get_full_path_or_raise("loras", lora2_file)

        if lora1_path != self.lora1_path or lora2_path != self.lora2_path:
            self.lora1_path = lora1_path
            self.lora2_path = lora2_path
            try:
                self.lora1 = comfy.utils.load_torch_file(lora1_path, safe_load=True)
                self.lora2 = comfy.utils.load_torch_file(lora2_path, safe_load=True)
            except Exception as e:
                print(f"Error loading LoRAs: {e}")
                return (model,)

        lora1_strength = 1.0 - blend_factor
        lora2_strength = blend_factor
        model_lora = model
        if self.lora1:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, None, self.lora1,lora1_strength, 1)
        if self.lora2:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, None, self.lora2, lora2_strength, 1)

        return (model_lora,)


NODE_CLASS_MAPPINGS = {
    "LoraBlend": LoraBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraBlend": "Lora Blend (Dropdown)",
}