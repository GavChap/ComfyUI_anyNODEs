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

### 2. LoRA XY Samplers
These nodes allow you to generate comparison grids for multiple LoRAs at varying strengths. They include an integrated sampling loop to make grid generation much faster than manual workflows.

#### **LoRA XY Integrated Sampler (Standard)**
Designed for standard workflows using a built-in KSampler logic.
- **Usage**: Plug in your model, clip, and conditioning. Select up to 10 LoRAs in the optional slots.
- **Strengths**: A multiline text box where you can list the strengths to test (e.g., `0.5`, `0.8`, `1.0`). Each LoRA will be tested at each strength.
- **Baseline**: Enable `include_baseline` to show a "No LoRA" image at the start of the grid.
- **Columns**: Control how many images appear per row in the final grid.

#### **LoRA XY Integrated Sampler (Custom)**
Designed for advanced workflows using the **Custom Sampler** infrastructure (Samplers/Sigmas/Noise).
- **Usage**: Same LoRA selection logic as the standard version, but allows you to pipe in custom `SAMPLER` and `SIGMAS` nodes.
- **Flexibility**: Ideal for use with specialized schedulers, PowerloRA, or complex noise injection workflows.

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
- **LoRA Nodes**: Found under the `anyMODE/batch` or `anyMODE/LoRA` categories.

## Cross-Platform Support
Internal label rendering uses a bundled **Roboto-Regular.ttf** font, ensuring that XY Grid labels render consistently across Windows and Linux without requiring system font installation.
