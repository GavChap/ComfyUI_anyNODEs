import os
import sys
import json
import torch
import folder_paths
from .quantization import utils, core
from safetensors.torch import save_file
from safetensors import safe_open

# extra.calibration is a ComfyUI-internal module that may not be directly
# importable depending on sys.path. Try multiple resolution strategies.
calibration = None
try:
    import extra.calibration as calibration
except ImportError:
    # Fall back: try importing relative to the ComfyUI base directory
    try:
        comfy_base = os.path.dirname(folder_paths.base_path) if hasattr(folder_paths, 'base_path') else None
        if comfy_base and comfy_base not in sys.path:
            sys.path.insert(0, comfy_base)
            import extra.calibration as calibration
    except (ImportError, AttributeError):
        pass

if calibration is None:
    print("[anyMODE] Warning: 'extra.calibration' module not found. Calibration nodes will be disabled.")
    print("[anyMODE] This module is part of a custom ComfyUI fork with calibration support.")

class AnyModeCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "filepath": ("STRING", {"default": "calibration/data.json"}),
                "keep_existing": ("BOOLEAN", {"default": True}),
                "clear_data": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "toggle"
    CATEGORY = "anyMODE/calibration"

    def toggle(self, model, enabled, filepath, keep_existing, clear_data):
        if calibration is None:
            print("[anyMODE] Calibration module not available. Cannot toggle calibration.")
            return (model,)

        # Handle path
        full_path = os.path.join(folder_paths.get_output_directory(), filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        calibration.set_enabled(enabled)
        
        if enabled:
            if keep_existing:
                calibration.load(full_path)
            calibration.set_save_path(full_path)
            print(f"Calibration enabled. Saving to: {full_path}")

        if clear_data:
            calibration.CALIB_DATA.clear()
            calibration.MODEL_SIGMA_RANGE.clear()
            print("Calibration data cleared.")
        
        return (model,)

class AnyModeSaveQuantizedWithCalibration:
    @classmethod
    def INPUT_TYPES(s):
        config_dir = os.path.join(os.path.dirname(__file__), "quantization", "configs")
        configs = ["custom"]
        if os.path.exists(config_dir):
            configs += sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])
            
        return {
            "required": {
                "model": ("MODEL",),
                "config_name": (configs, {"default": configs[1] if len(configs) > 1 else configs[0]}),
                "custom_config": ("STRING", {"multiline": True, "default": "{}"}),
                "method": (["mse", "amax", "percentile"], {"default": "mse"}),
                "n_samples": ("INT", {"default": 131072, "min": 0, "max": 1000000, "step": 1024}),
                "dense_search": ("BOOLEAN", {"default": False}),
                "stochastic_rounding": ("BOOLEAN", {"default": True}),
                "downcast_fp32": (["none", "fp16", "bf16"], {"default": "none"}),
                "save_path": ("STRING", {"default": "quantized/model_calibrated.safetensors"}),
                "calib_filepath": ("STRING", {"default": "calibration/data.json"}),
                "calibration_margin": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "anyMODE/calibration"

    def save(self, model, config_name, custom_config, method, n_samples, dense_search, stochastic_rounding, downcast_fp32, save_path, calib_filepath, calibration_margin):
        if calibration is None:
            print("[anyMODE] Calibration module not available. Cannot save calibrated model.")
            return {}

        device = utils.get_device()
        
        # Load from file if it exists to ensure we have the latest stats
        if calib_filepath:
            full_calib_path = os.path.join(folder_paths.get_output_directory(), calib_filepath)
            calibration.load(full_calib_path)

        calib_data = calibration.CALIB_DATA
        if not calib_data:
            print("Warning: No calibration data found!")
        
        # Parse amax
        processed_calib = {}
        for layer_bin, stats in calib_data.items():
            if "_bin" not in layer_bin: continue
            layer_name = layer_bin.rsplit("_bin", 1)[0]
            amax = stats.get("amax", 0)
            if layer_name not in processed_calib or amax > processed_calib[layer_name]:
                processed_calib[layer_name] = amax
        
        # Load Config
        if config_name == "custom":
            config = json.loads(custom_config)
        else:
            config_path = os.path.join(os.path.dirname(__file__), "quantization", "configs", config_name)
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        # Clone state dict to avoid mutating user's live memory state dict
        sd = {k: v.clone() for k, v in model.model.state_dict().items()}

        def get_qfmt(internal_key):
            return core.first_matching_qtype_for_key(internal_key, config.get("rules", []))
            
        prefixes = ["", "model.diffusion_model.", "model."]
        FP4_MAX = 6.0
        
        input_scale_dict = {}

        # 1. Math SmoothQuant migration (Scale Weights inversely prior to Quantization)
        for layer_name, amax_val in processed_calib.items():
            internal_key = layer_name + ".weight"
            
            # Find the true key in SD
            found_key = None
            for p in prefixes:
                if f"{p}{internal_key}" in sd:
                    found_key = f"{p}{internal_key}"
                    break
            
            if not found_key: continue

            qfmt = get_qfmt(internal_key)
            if not qfmt: continue

            val = float(amax_val) * calibration_margin
            
            # Calculate input scale
            if qfmt in ["nvfp4", "mxfp8", "float8_e4m3fn", "float8_e5m2"]:
                m = 448.0 if "e4m3" in qfmt or qfmt in ["nvfp4", "mxfp8"] else 57344.0
                if qfmt == "nvfp4":
                    input_scale_val = val / (m * FP4_MAX)
                else: 
                    input_scale_val = val / m
                    
                input_scale_dict[f"{found_key.rsplit('.weight', 1)[0]}.input_scale"] = input_scale_val
                
                # Pre-scale Weights (W_new = W / S)
                # Since activation will be X_new = X * S during inference
                sd[found_key] = sd[found_key] / input_scale_val
                print(f"Applied calibration scale {input_scale_val:.6f} to {layer_name}")

        # 2. Process State Dict to quantize
        print(f"Quantizing model using {config_name} with calibration injected...")
        new_sd, metadata = core.process_state_dict(sd, config, method, n_samples, downcast_fp32, device, stochastic=stochastic_rounding, dense_search=dense_search, verbose=True)

        # 3. Add original input scales dynamically into finalized layout
        for key, input_scale_val in input_scale_dict.items():
             new_sd[key] = torch.tensor(input_scale_val, dtype=torch.float32)

        # Metadata reconstruction
        save_metadata = metadata if metadata else {}
        
        # Try to gather model architecture
        original_meta = getattr(model.model, "metadata", getattr(model.model, "user_header", {}))
        for k, v in original_meta.items():
             if isinstance(v, str): save_metadata[k] = v
             else:
                 try: save_metadata[k] = json.dumps(v)
                 except: pass
                 
        if "architecture" not in save_metadata and "format" not in save_metadata:
            model_config = getattr(model.model, "model_config", None)
            if model_config:
                from comfy.supported_models import Flux, SDXL, SD15, SD21
                if isinstance(model_config, Flux): save_metadata["format"] = "flux"
                elif isinstance(model_config, SDXL): save_metadata["format"] = "sdxl"
        
        # 4. Save
        full_save_path = os.path.join(folder_paths.get_output_directory(), save_path)
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        print(f"Saving globally-calibrated Quantized model to {full_save_path}")
        save_file(new_sd, full_save_path, metadata=save_metadata)
        
        return {}

NODE_CLASS_MAPPINGS = {
    "AnyModeCalibration": AnyModeCalibration,
    "AnyModeSaveQuantizedWithCalibration": AnyModeSaveQuantizedWithCalibration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyModeCalibration": "anyMODE Calibration (Toggle)",
    "AnyModeSaveQuantizedWithCalibration": "anyMODE Save Calibrated Model"
}
