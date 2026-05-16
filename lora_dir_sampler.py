import torch
import math
import folder_paths
import comfy.utils
import comfy.sd
import nodes
import os
from .lora_xy_grid import draw_label

class LoraDirectorySampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_directory"
    CATEGORY = "anyMODE/batch"

    def sample_directory(self, model, clip, vae, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, lora_name, strength):
        if lora_name == "None":
            return (torch.zeros((1, 64, 64, 3)),)

        lora_full_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_full_path:
            raise Exception(f"LoRA not found: {lora_name}")
        
        lora_directory = os.path.dirname(lora_full_path)

        lora_files = [f for f in os.listdir(lora_directory) if f.lower().endswith(('.safetensors', '.ckpt', '.pt'))]
        lora_files.sort()
        
        if not lora_files:
            return (torch.zeros((1, 64, 64, 3)),)

        results = []
        total = len(lora_files)
        
        for idx, lora_file in enumerate(lora_files):
            print(f"anyMODE: Sampling {idx+1}/{total} - {lora_file}")
            lora_path = os.path.join(lora_directory, lora_file)
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            m, c = comfy.sd.load_lora_for_models(model, clip, lora_data, strength, strength)
            
            samples = nodes.common_ksampler(m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            images = vae.decode(samples[0]["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            
            labeled = draw_label(images.cpu(), f"{lora_file}\nS: {strength}")
            results.append(labeled)

        # Optimum packing for squarest collage
        _, h, w, ch = results[0].shape
        best_cols = 1
        min_diff = float('inf')
        for c in range(1, total + 1):
            r = math.ceil(total / c)
            diff = abs((c * w) - (r * h))
            if diff < min_diff:
                min_diff = diff
                best_cols = c
        
        cols, rows = best_cols, math.ceil(total / best_cols)
        grid = torch.zeros((1, h * rows, w * cols, ch))
        for i, img in enumerate(results):
            grid[0, (i // cols)*h:((i // cols)+1)*h, (i % cols)*w:((i % cols)+1)*w, :] = img[0]
            
        return (grid,)

NODE_CLASS_MAPPINGS = {
    "LoraDirectorySampler": LoraDirectorySampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraDirectorySampler": "LoRA Directory Sampler"
}
