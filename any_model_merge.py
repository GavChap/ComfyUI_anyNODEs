class AnyModelMerge10:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "normalize": ("BOOLEAN", {"default": False}),
            },
            "optional": {}
        }
        for i in range(1, 11):
            inputs["optional"][f"model_{i}"] = ("MODEL",)
            inputs["optional"][f"ratio_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
        return inputs
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "anyMODE/model_merge"

    def merge(self, normalize, **kwargs):
        models = []
        ratios = []
        for i in range(1, 11):
            model = kwargs.get(f"model_{i}")
            ratio = kwargs.get(f"ratio_{i}", 1.0)
            if model is not None:
                models.append(model)
                ratios.append(ratio)
        
        if len(models) == 0:
            return (None,)
        
        if normalize:
            total = sum(ratios)
            if total > 1.0:
                ratios = [r / total for r in ratios]
        
        m = models[0].clone()
        
        # Scale the first model by its ratio
        if ratios[0] != 1.0:
            kp0 = m.get_key_patches("diffusion_model.")
            for k in kp0:
                m.add_patches({k: kp0[k]}, 0.0, ratios[0])
        
        # Add subsequent models
        for i in range(1, len(models)):
            kpi = models[i].get_key_patches("diffusion_model.")
            for k in kpi:
                m.add_patches({k: kpi[k]}, ratios[i], 1.0)
                
        return (m,)
