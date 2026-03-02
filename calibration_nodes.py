import os
import json
import torch
import folder_paths
import extra.calibration as calibration
from .quantization import utils, core
from safetensors.torch import save_file
from safetensors import safe_open

class AnyModeCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "clear_data": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "toggle"
    CATEGORY = "anyMODE/calibration"

    def toggle(self, model, enabled, clear_data):
        calibration.set_enabled(enabled)
        if clear_data:
            calibration.CALIB_DATA.clear()
            calibration.MODEL_SIGMA_RANGE.clear()
            print("Calibration data cleared.")
        
        # We don't necessarily need to modify the model here, 
        # but if calibration is enabled, future samplings will record data.
        # However, to be sure the model is using the right ops, 
        # we might want to reload it or ensure it's using MixedPrecisionOps.
        
        # If the model was loaded while calibration was disabled, it might not have the hooks.
        # But comfy/ops.py checks calibration.ENABLED at runtime in forward().
        # So as long as it's using MixedPrecisionOps, it's fine.
        
        return (model,)

class AnyModeSaveQuantizedWithCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "save_path": ("STRING", {"default": "quantized/model_calibrated.safetensors"}),
                "calibration_margin": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "anyMODE/calibration"

    def save(self, model, save_path, calibration_margin):
        # This logic is adapted from add_input_scale.py
        
        calib_data = calibration.CALIB_DATA
        if not calib_data:
            print("Warning: No calibration data found!")
            # Should we still save? Probably better to error or warn.
        
        # Parse amax from calib_data
        # calibration.py stores data as layer_name_binX
        processed_calib = {}
        for layer_bin, stats in calib_data.items():
            if "_bin" not in layer_bin:
                continue
            layer_name = layer_bin.rsplit("_bin", 1)[0]
            amax = stats.get("amax", 0)
            if layer_name not in processed_calib or amax > processed_calib[layer_name]:
                processed_calib[layer_name] = amax
        
        # Get state dict from model
        sd = model.model.state_dict()
        
        # Get metadata
        # ComfyUI's model patcher doesn't easily expose the original safetensors metadata
        # unless we find it.
        metadata = getattr(model.model, "metadata", {})
        
        out_tensors = {}
        for k, v in sd.items():
            out_tensors[k] = v
        
        # Constants for scaling
        FP4_MAX = 6.0
        
        def get_qfmt(dt):
            s = str(dt).upper()
            if dt == torch.uint8 or "UINT8" in s:
                return "nvfp4"
            if dt == torch.float8_e4m3fn or "E4M3" in s:
                return "float8_e4m3fn"
            if dt == torch.float8_e5m2 or "E5M2" in s:
                return "float8_e5m2"
            return None

        def get_input_scale(qfmt, value):
            value = value * calibration_margin
            if qfmt == "nvfp4":
                fp8_max = 448.0 # F8_E4M3_MAX
                return value / (fp8_max * FP4_MAX)
            if qfmt in ("float8_e4m3fn", "float8_e5m2"):
                # Simplified max values if not importing ck.float_utils
                m = 448.0 if qfmt == "float8_e4m3fn" else 57344.0
                return value / m
            return None

        prefixes = ["", "model.diffusion_model.", "model."]
        
        for layer_name, amax_val in processed_calib.items():
            found_weight_key = None
            for p in prefixes:
                candidate = f"{p}{layer_name}.weight"
                if candidate in sd:
                    found_weight_key = candidate
                    break
            
            if not found_weight_key:
                continue
            
            weight = sd[found_weight_key]
            qfmt = get_qfmt(weight.dtype)
            if not qfmt:
                # Might be using comfy_quant metadata for format
                # Let's check if there is a comfy_quant tensor
                prefix = found_weight_key.rsplit(f"{layer_name}.weight", 1)[0]
                qconf_key = f"{prefix}{layer_name}.comfy_quant"
                if qconf_key in sd:
                    try:
                        qconf = json.loads(sd[qconf_key].numpy().tobytes())
                        qfmt = qconf.get("format")
                    except:
                        pass
            
            if not qfmt:
                continue
                
            input_scale_val = get_input_scale(qfmt, float(amax_val))
            if input_scale_val is not None:
                prefix = found_weight_key.rsplit(f"{layer_name}.weight", 1)[0]
                out_tensors[f"{prefix}{layer_name}.input_scale"] = torch.tensor(input_scale_val, dtype=torch.float32)
                
                # Log progress
                print(f"Added input_scale for {layer_name}: {input_scale_val:.6f}")

        full_save_path = os.path.join(folder_paths.get_output_directory(), save_path)
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        
        # Save
        # If there's original metadata, try to preserve it
        save_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, str):
                    save_metadata[k] = v
                else:
                    save_metadata[k] = json.dumps(v)

        save_file(out_tensors, full_save_path, metadata=save_metadata)
        print(f"Calibrated model saved to: {full_save_path}")
        
        return {}

NODE_CLASS_MAPPINGS = {
    "AnyModeCalibration": AnyModeCalibration,
    "AnyModeSaveQuantizedWithCalibration": AnyModeSaveQuantizedWithCalibration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyModeCalibration": "anyMODE Calibration (Toggle)",
    "AnyModeSaveQuantizedWithCalibration": "anyMODE Save Calibrated Model"
}
