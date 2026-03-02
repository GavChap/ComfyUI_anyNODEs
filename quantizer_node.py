import os
import json
import torch
from safetensors.torch import save_file
import folder_paths
from .quantization import core, utils

# Ensure Triton is enabled for Blackwell support
try:
    import comfy_kitchen
    comfy_kitchen.enable_backend("triton")
except:
    pass

class AnyModeQuantizer:
    @classmethod
    def INPUT_TYPES(s):
        config_dir = os.path.join(os.path.dirname(__file__), "quantization", "configs")
        configs = ["custom"]
        if os.path.exists(config_dir):
            configs += sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])
        
        return {
            "required": {
                "model": ("MODEL",),
                "config_name": (configs, {"default": "flux-2-klein-9b-nvfp4.json" if "flux-2-klein-9b-nvfp4.json" in configs else configs[0]}),
                "custom_config": ("STRING", {"multiline": True, "default": "{}"}),
                "method": (["mse", "amax", "percentile"], {"default": "mse"}),
                "n_samples": ("INT", {"default": 131072, "min": 0, "max": 1000000, "step": 1024}),
                "downcast_fp32": (["none", "fp16", "bf16"], {"default": "none"}),
                "save_model": ("BOOLEAN", {"default": False}),
                "save_path": ("STRING", {"default": "quantized/model.safetensors"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "quantize"
    CATEGORY = "anyMODE"

    def quantize(self, model, config_name, custom_config, method, n_samples, downcast_fp32, save_model, save_path):
        device = utils.get_device()
        
        # Load config
        if config_name == "custom":
            config = json.loads(custom_config)
        else:
            config_path = os.path.join(os.path.dirname(__file__), "quantization", "configs", config_name)
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        # Get state dict
        # We need the full state dict. ComfyUI's model patcher doesn't easily give the full state dict in one go
        # if it's not loaded. But usually, we can get it from model.model.state_dict().
        # However, it's better to use the model patcher's load_device if it's already there or loaded.
        
        print(f"Quantizing model using config: {config_name}")
        
        # Pull the state dict from the model patcher
        # Note: this might load the whole model into memory.
        sd = model.model.state_dict()
        
        # Process quantization
        new_sd, metadata = core.process_state_dict(sd, config, method, n_samples, downcast_fp32, device, verbose=True)
        
        # Save if requested
        if save_model:
            full_save_path = os.path.join(folder_paths.get_output_directory(), save_path)
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            print(f"Saving quantized model to {full_save_path}")
            save_file(new_sd, full_save_path, metadata=metadata)
        
        # Create a new model patcher with the quantized weights
        # We can try to patch the existing model or create a new one.
        # ComfyUI's ModelPatcher expects the weights to be in the state dict.
        # But we also need to make sure the model knows how to handle the quantized tensors.
        # ComfyUI's ops.py handles 'comfy_quant' metadata.
        
        new_model = model.clone()
        # This is a bit tricky. We want to replace the model's weights.
        # Actually, if we just update the model.model.load_state_dict(new_sd), it should work
        # but ModelPatcher might have its own ideas.
        
        # A safer way in ComfyUI is to create a new model object if possible, 
        # but here we can just replace the internal model weights.
        new_model.model.load_state_dict(new_sd, strict=False)
        
        return (new_model,)

NODE_CLASS_MAPPINGS = {
    "AnyModeQuantizer": AnyModeQuantizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyModeQuantizer": "anyMODE Quantizer"
}
