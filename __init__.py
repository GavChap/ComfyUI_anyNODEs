from .any_text_generate import AnyTextGenerate

NODE_CLASS_MAPPINGS = {
    "AnyTextGenerate": AnyTextGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyTextGenerate": "Any Text Generate (with System Prompt)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
