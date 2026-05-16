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

def generate_model_grid(model_1, clip, vae, latent_image, cfgs, columns, include_baseline, image_differences, diff_target, kwargs, sampler_func):
    selected_models = []
    
    model_labels_raw = kwargs.get("model_labels", "")
    model_labels = [label.strip() for label in model_labels_raw.split('\n') if label.strip()]

    label_1 = model_labels[0] if len(model_labels) > 0 else "Model 1"
    selected_models.append((model_1, label_1))

    # Collect models 2 to 10
    for i in range(2, 11):
        m = kwargs.get(f"model_{i}")
        if m is not None:
            label = model_labels[len(selected_models)] if len(selected_models) < len(model_labels) else f"Model {i}"
            selected_models.append((m, label))

    # cfgs parsing
    cfg_list = []
    for s in cfgs.replace(',', '\n').split('\n'):
        s = s.strip()
        if s:
            try:
                cfg_list.append(float(s))
            except ValueError:
                continue

    if not cfg_list:
        cfg_list.append(kwargs.get('base_cfg', 8.0))

    results = []
    baseline_image = None
    
    num_diffs = 0
    if image_differences != "none":
        if "baseline" in diff_target: num_diffs += 1
        if "previous" in diff_target: num_diffs += 1
        
    outputs_per_combo = num_diffs
    if image_differences in ["none", "both", "both magnified"]:
        outputs_per_combo += 1
        
    needs_baseline = include_baseline == "enable" or (image_differences != "none" and "baseline" in diff_target)
    
    if needs_baseline:
        print(f"anyMODE: Sampling baseline model")
        base_cfg = cfg_list[0]
        samples = sampler_func(model_1, base_cfg)
        images = vae.decode(samples[0]["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        baseline_image = images.cpu()
        
        if include_baseline == "enable":
            labeled_image = draw_label(baseline_image, "Baseline")
            results.append(labeled_image)
            for _ in range(outputs_per_combo - 1):
                spacer = torch.zeros_like(labeled_image)
                results.append(spacer)

    combinations = []
    for model_obj, label in selected_models:
        for c in cfg_list:
            combinations.append((model_obj, label, c))
            
    if not combinations and include_baseline == "disable":
        base_cfg = cfg_list[0]
        samples = sampler_func(model_1, base_cfg)
        images = vae.decode(samples[0]["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)

    total = len(combinations)
    previous_image = None
    
    for idx, (current_model, label, current_cfg) in enumerate(combinations):
        print(f"anyMODE: Sampling {idx+1}/{total} - {label} @ CFG {current_cfg}")
        
        samples = sampler_func(current_model, current_cfg)
        images = vae.decode(samples[0]["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        current_image = images.cpu()
        
        if image_differences in ["none", "both", "both magnified"]:
            label_text = f"{label}\nCFG: {current_cfg}"
            labeled_image = draw_label(current_image, label_text)
            results.append(labeled_image)
            
        if image_differences != "none":
            is_magnified = "magnified" in image_differences
            
            if "baseline" in diff_target and baseline_image is not None:
                diff = torch.abs(current_image[0:1] - baseline_image[0:1])
                if is_magnified:
                    diff = torch.clamp(diff * 5.0, 0.0, 1.0)
                
                diff_label = f"{label}\nCFG: {current_cfg} (-Base)"
                if is_magnified:
                    diff_label += " x5"
                labeled_diff = draw_label(diff, diff_label)
                results.append(labeled_diff)

            if "previous" in diff_target:
                if previous_image is not None:
                    diff_prev = torch.abs(current_image[0:1] - previous_image[0:1])
                else:
                    diff_prev = torch.zeros_like(current_image[0:1])
                
                if is_magnified:
                    diff_prev = torch.clamp(diff_prev * 5.0, 0.0, 1.0)
                
                diff_label_prev = f"{label}\nCFG: {current_cfg} (-Prev)"
                if is_magnified:
                    diff_label_prev += " x5"
                labeled_diff_prev = draw_label(diff_prev, diff_label_prev)
                results.append(labeled_diff_prev)
                
        previous_image = current_image

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
    return {
        "required": {
            "model_1": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
        },
        "optional": {
            "model_2": ("MODEL",),
            "model_3": ("MODEL",),
            "model_4": ("MODEL",),
            "model_5": ("MODEL",),
            "model_6": ("MODEL",),
            "model_7": ("MODEL",),
            "model_8": ("MODEL",),
            "model_9": ("MODEL",),
            "model_10": ("MODEL",),
            "model_labels": ("STRING", {"multiline": True, "default": "Model 1\nModel 2\nModel 3\nModel 4\nModel 5\nModel 6\nModel 7\nModel 8\nModel 9\nModel 10"}),
        }
    }


class ModelXYIntegratedSampler:
    @classmethod
    def INPUT_TYPES(s):
        inputs = get_base_input_types()
        inputs["required"].update({
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "cfgs": ("STRING", {"multiline": True, "default": "8.0"}),
            "columns": ("INT", {"default": 3, "min": 1, "max": 100}),
            "include_baseline": (["disable", "enable"], {"default": "disable"}),
            "image_differences": (["none", "both", "both magnified", "diff only", "diff magnified only"], {"default": "none"}),
            "diff_target": (["baseline", "previous", "baseline & previous"], {"default": "baseline"}),
        })
        return inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_grid"
    CATEGORY = "anyMODE/batch"

    def sample_grid(self, model_1, clip, vae, positive, negative, latent_image, seed, steps, sampler_name, scheduler, denoise, cfgs, columns, include_baseline, image_differences, diff_target="baseline", **kwargs):
        def sampler_func(model_to_use, cfg_to_use):
            return nodes.common_ksampler(model_to_use, seed, steps, cfg_to_use, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        
        kwargs['base_cfg'] = 8.0
        return generate_model_grid(model_1, clip, vae, latent_image, cfgs, columns, include_baseline, image_differences, diff_target, kwargs, sampler_func)


class ModelXYIntegratedSamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        inputs = get_base_input_types()
        inputs["required"].update({
            "add_noise": (["enable", "disable"],),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "cfgs": ("STRING", {"multiline": True, "default": "8.0"}),
            "columns": ("INT", {"default": 3, "min": 1, "max": 100}),
            "include_baseline": (["disable", "enable"], {"default": "disable"}),
            "image_differences": (["none", "both", "both magnified", "diff only", "diff magnified only"], {"default": "none"}),
            "diff_target": (["baseline", "previous", "baseline & previous"], {"default": "baseline"}),
        })
        return inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_grid"
    CATEGORY = "anyMODE/batch"

    def sample_grid(self, model_1, clip, vae, add_noise, noise_seed, sampler, sigmas, positive, negative, latent_image, cfgs, columns, include_baseline, image_differences, diff_target="baseline", **kwargs):
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        
        def sampler_func(model_to_use, cfg_to_use):
            return SamplerCustom().sample(model=model_to_use, add_noise=add_noise=="enable", noise_seed=noise_seed, cfg=cfg_to_use, positive=positive, negative=negative, sampler=sampler, sigmas=sigmas, latent_image=latent_image)

        kwargs['base_cfg'] = 8.0
        return generate_model_grid(model_1, clip, vae, latent_image, cfgs, columns, include_baseline, image_differences, diff_target, kwargs, sampler_func)

NODE_CLASS_MAPPINGS = {
    "ModelXYIntegratedSampler": ModelXYIntegratedSampler,
    "ModelXYIntegratedSamplerCustom": ModelXYIntegratedSamplerCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelXYIntegratedSampler": "Model XY Integrated Sampler",
    "ModelXYIntegratedSamplerCustom": "Model XY Integrated Sampler (Custom)",
}
