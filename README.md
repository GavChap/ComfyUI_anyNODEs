# ComfyUI_anyNODEs

A collection of specialized custom nodes for ComfyUI, designed to enhance text generation, LLM workflows, and LoRA management.

## Nodes Included

### 1. Any Text Generate (with System Prompt)
An enhanced version of the standard text generation node that provides more control over LLM interaction, particularly for reasoning models.

**Key Features:**
- **System Prompt Support**: Explicitly define a system instruction separate from the user message.
- **Thought Extraction**: Automatically detects `<think>...</think>` tags (used by models like DeepSeek-R1 or Gemma 2) and routes the internal reasoning to a dedicated **`thought`** output.
- **Thinking Toggle**: A `no_think` checkbox that appends `/no_think` to the prompt and cleans thinking tags from the result.
- **Vision Support**: Includes an optional `image` input for use with vision-capable models (e.g., LTX2).
- **Format-Aware**: Uses standardized prompt formatting (`<start_of_turn>system` / `<start_of_turn>user`) compatible with modern model architectures.

**Outputs:**
- `text`: The final cleaned response from the model.
- `thought`: The extracted reasoning process or "hidden" thinking from the model.

### 2. LoRA XY Integrated Sampler
A powerful node for generating XY grids to compare different LoRA strengths and combinations.
- Supports multiple LoRA slots with independent strength controls.
- Automatically generates labels for the resulting grid.
- Integrated sampling logic to streamline the comparison workflow.

### 3. Lora Blender
A utility node for blending multiple LoRAs together with weighted averages.
- Simplifies complex LoRA stacks into a single manageable output.
- Useful for fine-tuning the influence of multiple stylistic or character LoRAs.

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/GavChap/ComfyUI_anyNODEs.git
   ```
3. Restart ComfyUI.

## Usage
- **LLM Nodes**: Found under the `anyMODE/LLM` category.
- **LoRA Nodes**: Found under the `anyMODE/LoRA` or `anyMODE/Sampling` categories.
