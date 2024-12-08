import comfy
import folder_paths


class CheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),)
            }
        }

    RETURN_TYPES = ("MODEL_WEIGHTS", "CLIP_WEIGHTS", "VAE_WEIGHTS")
    FUNCTION = "load_checkpoint"

    CATEGORY = "slicer/loaders"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        sd = comfy.utils.load_torch_file(ckpt_path)

        unet_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        model_config = comfy.model_detection.model_config_from_unet(sd, unet_prefix)

        model_weights = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in unet_prefix}, filter_keys=True)

        vae_weights = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
        vae_weights = model_config.process_vae_state_dict(vae_weights)

        clip_weights = model_config.process_clip_state_dict(sd)

        return model_weights, clip_weights, vae_weights
