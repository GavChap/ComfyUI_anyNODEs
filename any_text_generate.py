import re

class AnyTextGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_length": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "sampling_mode": (["on", "off"], {"default": "on"}),
                "no_think": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thought")
    FUNCTION = "execute"
    CATEGORY = "anyMODE/LLM"

    def execute(self, clip, system_prompt, user_prompt, max_length, sampling_mode, no_think, temperature, top_k, top_p, min_p, repetition_penalty, seed, image=None):
        
        processed_user_prompt = user_prompt
        if no_think:
            processed_user_prompt = processed_user_prompt.strip() + " /no_think"

        # Format the prompt with system and user turns (Gemma style)
        if image is None:
            formatted_prompt = f"<start_of_turn>system\n{system_prompt.strip()}<end_of_turn>\n<start_of_turn>user\n{processed_user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # For image-to-text or vision-enhanced models
            formatted_prompt = f"<start_of_turn>system\n{system_prompt.strip()}<end_of_turn>\n<start_of_turn>user\n\n<image_soft_token>\n\n{processed_user_prompt}<end_of_turn>\n<start_of_turn>model\n"

        tokens = clip.tokenize(formatted_prompt, image=image, skip_template=False, min_length=1)

        do_sample = sampling_mode == "on"

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed
        )

        generated_text = clip.decode(generated_ids, skip_special_tokens=True)
        
        # Robust extraction of thought and response
        # 1. Try to find all closed <think>...</think> blocks
        thought_blocks = re.findall(r'<think>(.*?)</think>', generated_text, flags=re.DOTALL)
        
        # 2. Check for an unclosed <think> tag (common if max_length is reached)
        unclosed_think = ""
        if "<think>" in generated_text and "</think>" not in generated_text.split("<think>")[-1]:
             parts = generated_text.split("<think>")
             # The actual text is everything before the last <think>
             clean_text_base = "<think>".join(parts[:-1])
             # The unclosed thought is everything after the last <think>
             unclosed_think = parts[-1]
             
             # Remove other closed think blocks from clean_text_base
             clean_text = re.sub(r'<think>.*?</think>', '', clean_text_base, flags=re.DOTALL).strip()
        else:
             # Standard case: remove all closed blocks
             clean_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

        # Combine thoughts
        all_thoughts = []
        for block in thought_blocks:
            all_thoughts.append(block.strip())
        if unclosed_think:
            all_thoughts.append(unclosed_think.strip())
            
        thought_text = "\n\n[Thought Block]\n".join(filter(None, all_thoughts))
        
        return (clean_text, thought_text)

NODE_CLASS_MAPPINGS = {
    "AnyTextGenerate": AnyTextGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyTextGenerate": "Any Text Generate (with System Prompt)"
}
