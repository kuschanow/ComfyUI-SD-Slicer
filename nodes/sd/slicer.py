import comfy
import torch


class SdSlicer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_WEIGHTS",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "slice"

    OUTPUT_NODE = True

    CATEGORY = "slicer/sd"

    def slice(self, model):
        return model.get("", torch.Tensor)
