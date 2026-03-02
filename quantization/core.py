import torch
import json
import os
import comfy_kitchen as ck
from . import utils

def quantize_weight(weight, key, quantized_state_dict, quantization_layers, qtype, qformat, method, n_samples, device, verbose=True):
    layer_name = key[:-7] if key.endswith(".weight") else key
    
    if qtype == "nvfp4":
        if method == "mse":
            weight_scale_2 = utils.scale_mse_nvfp4(weight, n_samples=n_samples)
        else:
            weight_scale_2 = utils.scale_amax_nvfp4(weight)
        weight_quantized, weight_scale = ck.quantize_nvfp4(weight, weight_scale_2)
        if verbose: utils.print_layer_metrics(layer_name, weight, weight_quantized, weight_scale_2, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale_2"] = weight_scale_2.cpu()
    elif qtype == "mxfp8":
        orig_shape = tuple(weight.shape)
        weight_quantized, weight_scale = ck.quantize_mxfp8(weight)
        if verbose: print(f"MXFP8: {layer_name}")
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
    elif qtype == "int8_tensorwise":
        if method == "mse":
          weight_scale = utils.scale_mse_int8(weight, n_samples=n_samples)
        else:
          weight_scale = utils.scale_amax_int8(weight)
        weight_quantized = utils.quantize_per_tensor_int8(weight, weight_scale)
        if verbose: utils.print_layer_metrics(layer_name, weight, weight_quantized, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
    elif qtype == "int8_rowwise":
        if method == "percentile":
            weight_scales = utils.scale_rowwise_percentile_int8(weight)
        else:
            weight_scales = utils.scale_rowwise_amax_int8(weight)
        weight_quantized = utils.quantize_rowwise_int8(weight, weight_scales)
        if verbose: print(f"Int8 Rowwise: {layer_name}")
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scales.cpu()
    elif qtype == "float8_e5m2":
        # Placeholder for float8_e5m2 if needed, otherwise default to e4m3
        pass
    else: # fp8 e4m3
        if method == "mse":
            weight_scale = utils.scale_mse_fp8(weight, n_samples=n_samples)
        else:
            weight_scale = utils.scale_amax_fp8(weight)
        weight_quantized = ck.quantize_per_tensor_fp8(weight, weight_scale)
        if verbose: utils.print_layer_metrics(layer_name, weight, weight_quantized, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()

    qinfo = { "format": qtype }
    if qtype == "mxfp8":
        qinfo["orig_dtype"] = "torch.bfloat16"
        qinfo["orig_shape"] = orig_shape

    if qformat == "comfy_quant":
        quantized_state_dict[f"{layer_name}.comfy_quant"] = torch.tensor(
                list(json.dumps(qinfo).encode("utf-8")), dtype=torch.uint8)
    else:
        quantization_layers[layer_name] = qinfo

def store_with_optional_downcast(tensor, key, quantized_state_dict, cast_to, verbose=True):
    if tensor.dtype == torch.float32 and cast_to is not None:
        casted_weight = tensor.to(dtype=cast_to)
        quantized_state_dict[key] = casted_weight.cpu()
        if verbose and key.endswith(".weight"):
            layer_name = key[:-7]
            utils.print_layer_metrics(layer_name, tensor, casted_weight)
    else:
        quantized_state_dict[key] = tensor.cpu()

def first_matching_qtype_for_key(key, rules):
    for r in rules:
        matches = r.get("match", [])
        for p in matches:
            if p in key:
                qtype = r.get("policy")
                return qtype if qtype in utils.ALLOWED_QTYPES else None
    return None

def process_state_dict(state_dict, config, method, n_samples, downcast_fp32, device, verbose=True):
    cast_to = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(downcast_fp32, None)
    qformat = config.get("format", "comfy_quant")
    block_names = config.get("block_names", ["block", "transformer", "layer", "model.diffusion_model"])
    rules = config.get("rules", [])
    
    quantized_state_dict, quantization_layers = {}, {}
    
    if not verbose:
        utils.print_layer_header()

    for key, tensor in state_dict.items():
        original_key = key
        # Internal naming logic
        internal_key = key
        if internal_key.startswith("model.diffusion_model."):
            internal_key = internal_key[len("model.diffusion_model."):]
        elif internal_key.startswith("model."):
            internal_key = internal_key[len("model."):]

        is_quantizable = (any(b in internal_key for b in block_names) 
                          and internal_key.endswith(".weight")
                          and tensor.dtype in utils.QUANTIZABLE_WEIGHT_DTYPES 
                          and tensor.ndim == 2)

        if not is_quantizable:
            store_with_optional_downcast(tensor, original_key, quantized_state_dict, cast_to, verbose=verbose)
            continue

        qtype = first_matching_qtype_for_key(internal_key, rules)
        if qtype is None:
            store_with_optional_downcast(tensor, original_key, quantized_state_dict, cast_to, verbose=verbose)
        else:
            if verbose:
                print(f"Quantizing {original_key} as {qtype}")
            quantize_weight(tensor.to(device), original_key, quantized_state_dict, quantization_layers, qtype, qformat, method, n_samples, device, verbose=verbose)

    metadata = None
    if qformat != "comfy_quant":
         metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": quantization_layers})}
    
    return quantized_state_dict, metadata
