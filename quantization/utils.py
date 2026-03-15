import torch
import json
import os
import comfy_kitchen as ck
from comfy_kitchen.float_utils import F8_E4M3_MAX, F4_E2M1_MAX

# Constants
NUM_SAMPLE_DEFAULT = None
INT8_MAX = 127.0
ALLOWED_QTYPES = {"float8_e4m3fn", "float8_e5m2", "nvfp4", "mxfp8", "int8_tensorwise", "int8_rowwise"}
QUANTIZABLE_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Utility Functions from etc.py
def fixed_e(x, e=6, prec=4):
    if x is None: return "None"
    if isinstance(x, torch.Tensor): x = x.item()
    return f"{x * (10**e):.{prec}f}e-{e}"

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')

def get_metrics(original, quantized, global_scale=None, block_scales=None):
    if block_scales is not None and quantized.dtype == torch.uint8:
        assert global_scale is not None, "nvfp4 requires global_scale"
        dequantized = ck.dequantize_nvfp4(quantized, global_scale, block_scales, output_type=torch.float32)
    elif quantized.dtype == torch.float8_e4m3fn:
        assert global_scale is not None, "fp8 requires global_scale"
        dequantized = ck.dequantize_per_tensor_fp8(quantized, global_scale, output_type=torch.float32)
    elif quantized.dtype == torch.int8:
        assert global_scale is not None, "int8 requires global_scale"
        dequantized = dequantize_per_tensor_int8(quantized, global_scale)
    else:
        dequantized = quantized.to(dtype=torch.float32)

    original = original.to(torch.float32)
    amax = torch.amax(original.abs())
    mse = torch.mean((original - dequantized).pow(2))
    psnr = 10 * torch.log10(amax.pow(2) / (mse + 1e-10)) if mse > 0 else torch.tensor(torch.inf)

    signal_power = torch.mean(original.pow(2))
    sqnr = 10 * torch.log10(signal_power / (mse + 1e-10)) if mse > 0 else torch.tensor(torch.inf)

    max_err = torch.max(torch.abs(original - dequantized))
    rel_max_err = (max_err / (original.abs().amax() + 1e-8))

    orig_flat = original.flatten()
    dequant_flat = dequantized.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_flat.unsqueeze(0), dequant_flat.unsqueeze(0)
    )

    return mse.item(), sqnr.item(), psnr.item(), cos_sim.item(), max_err.item(), rel_max_err.item(), amax.item()

def print_layer_header():
    print(f"{'layer_name':-^35} {'dtype':-^10} {'scale':-^10} {'mse':-^10} {'psnr':-^7} {'sqnr':-^7} {'cos_sim':-^8} {'relmaxerr':-^8}")

def print_layer_metrics(layer_name, original, quantized, global_scale=None, block_scales=None):
    mse, sqnr, psnr, cos_sim, max_err, rel_max_err, amax = get_metrics(original, quantized, global_scale, block_scales)
    gs = f"{fixed_e(global_scale, 4, 3):>10}" if global_scale is not None else f"{'':>10}"
    print(f"{layer_name:<35} {str(quantized.dtype).partition('.')[2][:10]:>10} {gs} {fixed_e(mse, 6, 3):>10} {psnr:>6.4f} {sqnr:>6.4f} {cos_sim*100:8.4f} {rel_max_err*100:>8.4f}")

# Utility Functions from quant_method.py
def quantize_per_tensor_int8(x, scale):
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def dequantize_per_tensor_int8(x, scale):
    return x.float() * scale

def quantize_rowwise_int8(x, scales):
    x_float = x.float()
    # scales shape: (rows,) -> (rows, 1) for broadcasting
    inv_scales = (1.0 / scales).unsqueeze(-1)
    return (x_float * inv_scales).round().clamp(-128, 127).to(torch.int8)

def dequantize_rowwise_int8(x, scales):
    return x.float() * scales.unsqueeze(-1)

# Utility Functions from scale_search.py
def sample_flat(w, n, include_absmax=True):
    if w.numel() == 0:
        return w.new_empty((0,), dtype=torch.float32)
    x = w.flatten()
    if x.numel() <= n:
        return x.float()
    k = max(0, n - (1 if include_absmax else 0))
    idx = torch.randint(x.numel(), (k,), device=x.device)
    if include_absmax:
        idx = torch.cat([idx, x.abs().argmax().view(1)])
    return x[idx].float()

def sample_block16(w, n, include_absmax=True):
    if w.numel() == 0:
        return w.new_empty((0, 16), dtype=torch.float32)
    w = w.contiguous()
    r, c = w.shape
    assert r % 16 == 0 and c % 16 == 0, f"Rows and columns must be divisible by 16, got shape ({r}, {c})"
    x16 = w.view(r, c // 16, 16)
    n = ((n + 255) // 256) * 256
    n_blocks = n // 16
    total_blocks = x16.shape[0] * x16.shape[1]
    if total_blocks <= n_blocks:
        return x16.reshape(-1, 16).float()
    k = n_blocks - (1 if include_absmax else 0)
    ridx = torch.randint(x16.shape[0], (k,), device=w.device)
    bidx = torch.randint(x16.shape[1], (k,), device=w.device)
    if include_absmax:
        flat_pos = w.abs().view(-1).argmax().item()
        ar, ac = divmod(flat_pos, c)
        ridx = torch.cat([ridx, w.new_tensor([ar], dtype=torch.long)])
        bidx = torch.cat([bidx, w.new_tensor([ac // 16], dtype=torch.long)])
    return x16[ridx, bidx].float()

def scale_mse_nvfp4(w, n_samples=NUM_SAMPLE_DEFAULT, ratios=(0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10)):
    x = sample_block16(w, n_samples) if n_samples is not None else w.float()
    if x.numel() == 0: return w.new_tensor(0.0, dtype=torch.float32)
    amax = torch.amax(x.abs())
    if amax.item() == 0.0: return w.new_tensor(0.0, dtype=torch.float32)
    base = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
    best_scale, best_mse = base, float("inf")
    for r in ratios:
        scale = base * r
        quant, block_scales = ck.quantize_nvfp4(x, scale)
        dequant = ck.dequantize_nvfp4(quant, scale, block_scales, output_type=torch.float32)
        mse = (x - dequant).pow(2).mean().item()
        if mse < best_mse:
            best_scale, best_mse = scale, mse
    return best_scale.to(dtype=torch.float32)

def scale_mse_fp8(w, n_samples=NUM_SAMPLE_DEFAULT, ratios=(0.95, 0.975, 1.0, 1.025, 1.05)):
    x = sample_flat(w, n_samples) if n_samples is not None else w.float()
    if x.numel() == 0: return w.new_tensor(0.0, dtype=torch.float32)
    amax = torch.amax(x.abs())
    if amax.item() == 0.0: return w.new_tensor(0.0, dtype=torch.float32)
    base = amax / F8_E4M3_MAX
    best_scale, best_mse = base, float("inf")
    for r in ratios:
        scale = base * r
        quant = ck.quantize_per_tensor_fp8(x, scale, output_type=torch.float8_e4m3fn)
        dequant = ck.dequantize_per_tensor_fp8(quant, scale, output_type=torch.float32)
        mse = (x - dequant).pow(2).mean().item()
        if mse < best_mse:
            best_scale, best_mse = scale, mse
    return best_scale.to(dtype=torch.float32)

INT8_RATIOS = tuple(0.7 + 0.02*i for i in range(16))
def scale_mse_int8(w, n_samples=NUM_SAMPLE_DEFAULT, ratios=INT8_RATIOS):
    x = sample_flat(w, n_samples) if n_samples is not None else w.float()
    if x.numel() == 0: return w.new_tensor(0.0, dtype=torch.float32)
    amax = torch.amax(x.abs())
    if amax.item() == 0.0: return w.new_tensor(0.0, dtype=torch.float32)
    base = amax / INT8_MAX
    best_scale, best_mse = base, float("inf")
    for r in ratios:
        scale = base * r
        quant = quantize_per_tensor_int8(x, scale)
        dequant = dequantize_per_tensor_int8(quant, scale)
        mse = (x - dequant).pow(2).mean().item()
        if mse < best_mse:
            best_scale, best_mse = scale, mse
    return best_scale.to(dtype=torch.float32)

def scale_percentile(w, max_value, percentile=0.99999):
    return torch.quantile(w.abs().flatten().float(), percentile) / max_value

def scale_percentile_int8(w, percentile=0.99999):
    return scale_percentile(w, INT8_MAX, percentile)

def scale_amax(w, max_value):
    return torch.amax(w.abs()).to(dtype=torch.float32) / max_value

def scale_amax_nvfp4(w):
    return scale_amax(w, F8_E4M3_MAX * F4_E2M1_MAX)

def scale_amax_fp8(w):
    return scale_amax(w, F8_E4M3_MAX)

def scale_amax_int8(w):
    return scale_amax(w, INT8_MAX)

def scale_rowwise_percentile(w, max_value, percentile=0.99999):
    return torch.quantile(w.abs().float(), percentile, dim=1) / max_value

def scale_rowwise_percentile_int8(w, percentile=0.99999):
    return scale_rowwise_percentile(w, INT8_MAX, percentile)

def scale_rowwise_amax(w, max_value):
    return w.abs().amax(dim=1).to(torch.float32) / max_value

def scale_rowwise_amax_int8(w):
    return scale_rowwise_amax(w, INT8_MAX)
