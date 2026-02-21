import torch
import torch.nn.functional as F
import math
import folder_paths
import comfy.utils
import comfy.sd
import comfy.sample
import nodes
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Font loading optimization
_GRID_FONT = None
def get_grid_font():
    global _GRID_FONT
    if _GRID_FONT is not None:
        return _GRID_FONT
    
    # Prioritize bundled font
    bundled_font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Roboto-Regular.ttf")
    
    font_paths = [
        bundled_font_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    for p in font_paths:
        try:
            _GRID_FONT = ImageFont.truetype(p, 20)
            break
        except:
            continue
    if _GRID_FONT is None:
        try:
            _GRID_FONT = ImageFont.load_default()
        except:
            pass
    return _GRID_FONT

def draw_label(image_tensor, text):
    if not text:
        return image_tensor
    image_np = (image_tensor[0].numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)
    
    font = get_grid_font()

    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    except:
        left, top, right, bottom = 0, 0, 100, 20
        
    text_w = right - left
    text_h = bottom - top
    
    padding = 5
    rect_x1 = 0
    rect_y1 = pil_img.height - text_h - (padding * 2)
    rect_x2 = text_w + (padding * 2)
    rect_y2 = pil_img.height
    
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=(0, 0, 0))
    draw.text((padding, rect_y1 + padding), text, font=font, fill=(255, 255, 255))
    
    labeled_np = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(labeled_np).unsqueeze(0)

def generate_lora_grid(model, clip, vae, latent_image, strengths, columns, include_baseline, kwargs, sampler_func):
    """
    Helper function to generate the grid.
    sampler_func signature: sampler_func(model_lora) -> samples
    """
    selected_loras = []
    for i in range(1, 11):
        lora_val = kwargs.get(f"lora_{i}", "None")
        if lora_val != "None":
            selected_loras.append(lora_val)
    
    strength_list = []
    for s in strengths.replace(',', '\n').split('\n'):
        s = s.strip()
        if s:
            try:
                strength_list.append(float(s))
            except ValueError:
                continue
    
    if not selected_loras or not strength_list:
        if include_baseline == "disable":
            samples = sampler_func(model)
            images = vae.decode(samples[0]["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return (images,)
        # If strengths/loras are empty but baseline is enabled, we'll continue to generate just the baseline grid

    results = []
    if include_baseline == "enable":
        print(f"anyMODE: Sampling baseline (no LoRAs)")
        samples = sampler_func(model)
        images = vae.decode(samples[0]["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        labeled_image = draw_label(images.cpu(), "Baseline\n(No LoRAs)")
        results.append(labeled_image)

    combinations = []
    for lora in selected_loras:
        for s in strength_list:
            combinations.append((lora, s))
            
    if not combinations and include_baseline == "disable":
        samples = sampler_func(model)
        images = vae.decode(samples[0]["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)

    total = len(combinations)
    
    # Preload LoRA data
    unique_loras = list(set(selected_loras))
    lora_cache = {}
    for lora_name in unique_loras:
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path:
            lora_cache[lora_name] = comfy.utils.load_torch_file(lora_path, safe_load=True)
        else:
            lora_cache[lora_name] = None
    
    for idx, (lora_name, strength) in enumerate(combinations):
        print(f"anyMODE: Sampling {idx+1}/{total} - {lora_name} @ {strength}")
        lora_data = lora_cache.get(lora_name)
        if lora_data is not None:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_data, strength, strength)
        else:
            model_lora, clip_lora = model, clip
        
        samples = sampler_func(model_lora)
        images = vae.decode(samples[0]["samples"])
        if len(images.shape) == 5: # Combine batches/frames for grid
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        label_text = f"{os.path.basename(lora_name)}\nS: {strength}"
        labeled_image = draw_label(images.cpu(), label_text)
        results.append(labeled_image)

    if not results:
        return (torch.zeros((1, 64, 64, 3)),)
        
    first_img = results[0]
    batch_size, h, w, c = first_img.shape
    total_images = len(results)
    rows = math.ceil(total_images / columns)
    full_grid = torch.zeros((1, h * rows, w * columns, c))
    
    for idx, img in enumerate(results):
        row = idx // columns
        col = idx % columns
        if img.shape[1] != h or img.shape[2] != w:
            img_reshaped = img.permute(0, 3, 1, 2)
            img_resized = F.interpolate(img_reshaped, size=(h, w), mode='bilinear', align_corners=False)
            img = img_resized.permute(0, 2, 3, 1)
        full_grid[0, row*h:(row+1)*h, col*w:(col+1)*w, :] = img[0]

    return (full_grid,)

def get_base_input_types():
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    return {
        "required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
        },
        "optional": {
            "lora_1": (lora_list, {"default": "None"}),
            "lora_2": (lora_list, {"default": "None"}),
            "lora_3": (lora_list, {"default": "None"}),
            "lora_4": (lora_list, {"default": "None"}),
            "lora_5": (lora_list, {"default": "None"}),
            "lora_6": (lora_list, {"default": "None"}),
            "lora_7": (lora_list, {"default": "None"}),
            "lora_8": (lora_list, {"default": "None"}),
            "lora_9": (lora_list, {"default": "None"}),
            "lora_10": (lora_list, {"default": "None"}),
        }
    }


class LoraXYIntegratedSampler:
    @classmethod
    def INPUT_TYPES(s):
        inputs = get_base_input_types()
        inputs["required"].update({
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "strengths": ("STRING", {"multiline": True, "default": "1.0"}),
            "columns": ("INT", {"default": 3, "min": 1, "max": 100}),
            "include_baseline": (["disable", "enable"], {"default": "disable"}),
        })
        return inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_grid"
    CATEGORY = "anyMODE/batch"

    def sample_grid(self, model, clip, vae, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, strengths, columns, include_baseline, **kwargs):
        def sampler_func(model_lora):
            return nodes.common_ksampler(model_lora, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        
        return generate_lora_grid(model, clip, vae, latent_image, strengths, columns, include_baseline, kwargs, sampler_func)

class LoraXYIntegratedSamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        inputs = get_base_input_types()
        inputs["required"].update({
            "add_noise": (["enable", "disable"],),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "strengths": ("STRING", {"multiline": True, "default": "1.0"}),
            "columns": ("INT", {"default": 3, "min": 1, "max": 100}),
            "include_baseline": (["disable", "enable"], {"default": "disable"}),
        })
        return inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_grid"
    CATEGORY = "anyMODE/batch"

    def sample_grid(self, model, clip, vae, add_noise, noise_seed, cfg, sampler, sigmas, positive, negative, latent_image, strengths, columns, include_baseline, **kwargs):
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        
        def sampler_func(model_lora):
            return SamplerCustom().sample(model=model_lora, add_noise=add_noise=="enable", noise_seed=noise_seed, cfg=cfg, positive=positive, negative=negative, sampler=sampler, sigmas=sigmas, latent_image=latent_image)

        return generate_lora_grid(model, clip, vae, latent_image, strengths, columns, include_baseline, kwargs, sampler_func)

NODE_CLASS_MAPPINGS = {
    "LoraXYIntegratedSampler": LoraXYIntegratedSampler,
    "LoraXYIntegratedSamplerCustom": LoraXYIntegratedSamplerCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraXYIntegratedSampler": "LoRA XY Integrated Sampler",
    "LoraXYIntegratedSamplerCustom": "LoRA XY Integrated Sampler (Custom)",
}
