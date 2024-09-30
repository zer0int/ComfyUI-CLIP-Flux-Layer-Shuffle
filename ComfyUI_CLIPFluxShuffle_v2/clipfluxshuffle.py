import os
import torch
from torch import Tensor, nn
import time 
import copy
import importlib
import folder_paths
import comfy
import comfy.model_management as mm
import comfy.diffusers_load
import comfy.sd
import comfy.utils
import node_helpers
import comfy.clip_model
import comfy.supported_models_base
from comfy.model_management import get_torch_device

class CLIPshuffleBase:
    def __init__(self):
        super().__init__()
        self.original_clip_l = None
        self.original_t5xxl = None

    def get_original_models(self, clip):
        # Extract CLIP and T5XXL models from cond_stage_model
        clip_l = clip.cond_stage_model.clip_l
        t5xxl = clip.cond_stage_model.t5xxl

        # Deepcopy the original models if not already done
        if self.original_clip_l is None and self.original_t5xxl is None:
            self.original_clip_l = copy.deepcopy(clip_l)
            self.original_t5xxl = copy.deepcopy(t5xxl)
            timestamp = time.time()
            print(f"Original CLIP and T5 models saved at: {timestamp}")
        torch.cuda.empty_cache()
        # Return the first deepcopies (deepcopy I)
        return self.original_clip_l, self.original_t5xxl

class FluxShuffleBase:
    def __init__(self):
        super().__init__()
        self.original_model = None

    def get_original_model(self, model):
        if self.original_model is None:
            self.original_model = copy.deepcopy(model)
            timestamp = time.time()
            print(f"Original model saved at: {timestamp}")
        torch.cuda.empty_cache()
        timestamp = time.time()
        print(f"Model re-load invoked: {timestamp}")
        if hasattr(self.original_model, 'model'):
            themodel = self.original_model.model
            
        torch.cuda.empty_cache()
        model = copy.deepcopy(themodel)
        print("\n---------\nFIRST LOAD\n-----------")
        return model

class CLIPshuffleLayersNode(CLIPshuffleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "shuffle_setting_clip_attn": (["None", "Shuffle"],),
                "clip_attn_layers": ("STRING", {"default": ""}),
                "shuffle_setting_clip_mlp": (["None", "Shuffle"],),
                "clip_mlp_layers": ("STRING", {"default": ""}),
                "shuffle_setting_t5_attn": (["None", "Shuffle"],),
                "t5_attn_layers": ("STRING", {"default": ""}),
                "shuffle_setting_t5_mlp": (["None", "Shuffle"],),
                "t5_mlp_layers": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "shuffle_clip_t5"
    CATEGORY = "CLIP-Flux-Shuffle"

    def shuffle_clip_t5(self, clip, shuffle_setting_clip_attn, clip_attn_layers, shuffle_setting_clip_mlp, clip_mlp_layers, shuffle_setting_t5_attn, t5_attn_layers, shuffle_setting_t5_mlp, t5_mlp_layers):
        # Unwrap the CLIP and T5XXL models from the clip object
        original_clip_l, original_t5_model = self.get_original_models(clip)

        # Create new deepcopies from the first deepcopies (deepcopy II)
        clip_l = copy.deepcopy(original_clip_l)
        t5_model = copy.deepcopy(original_t5_model)
        torch.cuda.empty_cache()
        
        # Parsing layers from string input
        clip_attn_layers_to_shuffle = [int(x.strip()) for x in clip_attn_layers.split(",") if x.strip().isdigit()]
        clip_mlp_layers_to_shuffle = [int(x.strip()) for x in clip_mlp_layers.split(",") if x.strip().isdigit()]
        t5_attn_layers_to_shuffle = [int(x.strip()) for x in t5_attn_layers.split(",") if x.strip().isdigit()]
        t5_mlp_layers_to_shuffle = [int(x.strip()) for x in t5_mlp_layers.split(",") if x.strip().isdigit()]

        # Modify CLIP model layers
        if shuffle_setting_clip_attn == "Shuffle":
            for idx in range(len(clip_l.transformer.text_model.encoder.layers) - 2): # Avoid going out of range
                if idx in clip_attn_layers_to_shuffle:
                    current_block = clip_l.transformer.text_model.encoder.layers[idx]
                    next_block = clip_l.transformer.text_model.encoder.layers[idx + 1]
                    print(f"Shuffling CLIP attention layers: {idx} and {idx + 1}")
                    current_block.self_attn.q_proj.weight, next_block.self_attn.q_proj.weight = next_block.self_attn.q_proj.weight, current_block.self_attn.q_proj.weight

        if shuffle_setting_clip_mlp == "Shuffle":
            for idx in range(len(clip_l.transformer.text_model.encoder.layers) - 2): # Avoid going out of range
                if idx in clip_mlp_layers_to_shuffle:
                    current_block = clip_l.transformer.text_model.encoder.layers[idx]
                    next_block = clip_l.transformer.text_model.encoder.layers[idx + 1]
                    print(f"Shuffling CLIP MLP layers: {idx} and {idx + 1}")
                    current_block.mlp.fc1.weight, next_block.mlp.fc1.weight = next_block.mlp.fc1.weight, current_block.mlp.fc1.weight

        # Modify T5 model layers
        if shuffle_setting_t5_attn == "Shuffle":
            for idx in range(len(t5_model.transformer.encoder.block) - 2): # Avoid going out of range
                if idx in t5_attn_layers_to_shuffle:
                    current_block = t5_model.transformer.encoder.block[idx]
                    next_block = t5_model.transformer.encoder.block[idx + 1]
                    print(f"Shuffling T5 attention layers: {idx} and {idx + 1}")
                    current_block.layer[0].SelfAttention.q.weight, next_block.layer[0].SelfAttention.q.weight = next_block.layer[0].SelfAttention.q.weight, current_block.layer[0].SelfAttention.q.weight

        if shuffle_setting_t5_mlp == "Shuffle":
            for idx in range(len(t5_model.transformer.encoder.block) - 2): # Avoid going out of range
                if idx in t5_mlp_layers_to_shuffle:
                    current_block = t5_model.transformer.encoder.block[idx]
                    next_block = t5_model.transformer.encoder.block[idx + 1]
                    print(f"Shuffling T5 MLP layers: {idx} and {idx + 1}")
                    current_block.layer[1].DenseReluDense.wi_0.weight, next_block.layer[1].DenseReluDense.wi_0.weight = next_block.layer[1].DenseReluDense.wi_0.weight, current_block.layer[1].DenseReluDense.wi_0.weight

        # Re-wrap the modified models back into the FluxClipModel_ structure
        clip.cond_stage_model.clip_l = clip_l
        clip.cond_stage_model.t5xxl = t5_model

        return (clip,)

class ShuffleFluxLayersNode(FluxShuffleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), 
                "shuffle_setting_double": (["None", "Skip"],),
                "double_layers": ("STRING", {"default": ""}),               
                "shuffle_setting_attn_img": (["None", "Shuffle"],),
                "attn_layers_img": ("STRING", {"default": ""}),
                "shuffle_setting_attn_txt": (["None", "Shuffle"],),
                "attn_layers_txt": ("STRING", {"default": ""}),
                "shuffle_setting_attn_txt_proj": (["None", "Shuffle"],),
                "attn_layers_txtproj": ("STRING", {"default": ""}),
                "shuffle_setting_img_mlp0": (["None", "Shuffle"],),
                "img_mlp0_layers": ("STRING", {"default": ""}),
                "shuffle_setting_img_mlp2": (["None", "Shuffle"],),
                "img_mlp2_layers": ("STRING", {"default": ""}),              
                "shuffle_setting_txt_mlp0": (["None", "Shuffle"],),
                "txt_mlp0_layers": ("STRING", {"default": ""}),
                "shuffle_setting_txt_mlp2": (["None", "Shuffle"],),
                "txt_mlp2_layers": ("STRING", {"default": ""}),                
                "shuffle_setting_img_mod": (["None", "Shuffle"],),
                "img_mod_layers": ("STRING", {"default": ""}),
                "shuffle_setting_txt_mod": (["None", "Shuffle"],),
                "txt_mod_layers": ("STRING", {"default": ""}),                
                "shuffle_setting_imgnorm1": (["None", "Identity"],),
                "shuffle_setting_imgnorm2": (["None", "Identity"],),
                "shuffle_setting_txtnorm1": (["None", "Identity"],),
                "shuffle_setting_txtnorm2": (["None", "Identity"],),
                "all_identity": ("STRING", {"default": ""}), 
                "shuffle_setting_single": (["None", "Skip"],),
                "single_layers": ("STRING", {"default": ""}),
                "shuffle_setting_lin1": (["None", "Shuffle"],),
                "lin1_layers": ("STRING", {"default": ""}),
                "shuffle_setting_lin2": (["None", "Shuffle"],),
                "lin2_layers": ("STRING", {"default": ""}),   
                "shuffle_setting_prenorm": (["None", "Shuffle"],),
                "prenorm_layers": ("STRING", {"default": ""}),
                "shuffle_setting_sinmod": (["None", "Shuffle"],),
                "sinmod_layers": ("STRING", {"default": ""}),                
            }
        }
    CATEGORY = "CLIP-Flux-Shuffle"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "shuffle_and_apply"
   
    def shuffle_and_apply(self, model, shuffle_setting_double, double_layers, shuffle_setting_attn_img, attn_layers_img, shuffle_setting_attn_txt, attn_layers_txt, shuffle_setting_attn_txt_proj, attn_layers_txtproj, shuffle_setting_img_mlp0, img_mlp0_layers, shuffle_setting_img_mlp2, img_mlp2_layers, shuffle_setting_img_mod, img_mod_layers, shuffle_setting_txt_mod, txt_mod_layers, shuffle_setting_single, single_layers, shuffle_setting_lin1, lin1_layers, shuffle_setting_lin2, lin2_layers, shuffle_setting_prenorm, prenorm_layers, shuffle_setting_sinmod, sinmod_layers, shuffle_setting_txt_mlp0, txt_mlp0_layers, shuffle_setting_txt_mlp2, txt_mlp2_layers, shuffle_setting_txtnorm1, shuffle_setting_txtnorm2, all_identity, shuffle_setting_imgnorm1, shuffle_setting_imgnorm2):

        model = self.get_original_model(model)       
        diffusion_model = model.diffusion_model
        offload_device = mm.unet_offload_device()
        load_device = get_torch_device()     
        torch.cuda.empty_cache()
        
        # Parse layer input strings for each shuffling option
        double_layers_to_shuffle = [int(x.strip()) for x in double_layers.split(",") if x.strip().isdigit()]
        attn_layers_img_to_shuffle = [int(x.strip()) for x in attn_layers_img.split(",") if x.strip().isdigit()]
        attn_layers_txt_to_shuffle = [int(x.strip()) for x in attn_layers_txt.split(",") if x.strip().isdigit()]
        attn_layers_txtproj_to_shuffle = [int(x.strip()) for x in attn_layers_txtproj.split(",") if x.strip().isdigit()]
        img_mlp0_layers_to_shuffle = [int(x.strip()) for x in img_mlp0_layers.split(",") if x.strip().isdigit()]
        img_mlp2_layers_to_shuffle = [int(x.strip()) for x in img_mlp2_layers.split(",") if x.strip().isdigit()]
        txt_mlp0_layers_to_shuffle = [int(x.strip()) for x in txt_mlp0_layers.split(",") if x.strip().isdigit()]
        txt_mlp2_layers_to_shuffle = [int(x.strip()) for x in txt_mlp2_layers.split(",") if x.strip().isdigit()]      
        img_mod_layers_to_shuffle = [int(x.strip()) for x in img_mod_layers.split(",") if x.strip().isdigit()]
        txt_mod_layers_to_shuffle = [int(x.strip()) for x in txt_mod_layers.split(",") if x.strip().isdigit()]
        single_layers_to_shuffle = [int(x.strip()) for x in single_layers.split(",") if x.strip().isdigit()]
        lin1_layers_to_shuffle = [int(x.strip()) for x in lin1_layers.split(",") if x.strip().isdigit()]
        lin2_layers_to_shuffle = [int(x.strip()) for x in lin2_layers.split(",") if x.strip().isdigit()]
        prenorm_layers_to_shuffle = [int(x.strip()) for x in prenorm_layers.split(",") if x.strip().isdigit()]
        sinmod_layers_to_shuffle = [int(x.strip()) for x in sinmod_layers.split(",") if x.strip().isdigit()]
        all_identity_to_shuffle = [int(x.strip()) for x in all_identity.split(",") if x.strip().isdigit()]


        # Shuffle double block components based on the user's settings for each component
        for idx in range(len(diffusion_model.double_blocks) - 2):  # Avoid going out of range       
            current_block = diffusion_model.double_blocks[idx]
            next_block = diffusion_model.double_blocks[idx + 1]        
            
            if shuffle_setting_double == "Skip" and idx in double_layers_to_shuffle:
                # Find the next target layer in the skip list that comes after the current one
                subsequent_layers = [i for i in double_layers_to_shuffle if i > idx]

                if subsequent_layers:
                    # Get the first subsequent layer after the current one to connect to
                    target_idx = min(subsequent_layers)
                    target_block = diffusion_model.double_blocks[target_idx]                   
                    # Bypass intermediate layers: directly connect the current layer's output to the target layer's input
                    current_block.forward = lambda *args, **kwargs: target_block.forward(*args, **kwargs)
                else:
                    # If no subsequent layers to skip to, just proceed as normal
                    print(f"No subsequent layers to skip after layer {idx}. Proceeding normally.")

            if shuffle_setting_attn_img == "Shuffle" and idx in attn_layers_img_to_shuffle:
                current_block.img_attn.qkv.weight, next_block.img_attn.qkv.weight = next_block.img_attn.qkv.weight, current_block.img_attn.qkv.weight
                current_block.img_attn.qkv.bias, next_block.img_attn.qkv.bias = next_block.img_attn.qkv.bias, current_block.img_attn.qkv.bias
            
            if shuffle_setting_attn_txt == "Shuffle" and idx in attn_layers_txt_to_shuffle:
                current_block.txt_attn.qkv.weight, next_block.txt_attn.qkv.weight = next_block.txt_attn.qkv.weight, current_block.txt_attn.qkv.weight
                current_block.txt_attn.qkv.bias, next_block.txt_attn.qkv.bias = next_block.txt_attn.qkv.bias, current_block.txt_attn.qkv.bias

            if shuffle_setting_attn_txt_proj == "Shuffle" and idx in attn_layers_txtproj_to_shuffle:
                current_block.txt_attn.proj.weight, next_block.txt_attn.proj.weight = next_block.txt_attn.proj.weight, current_block.txt_attn.proj.weight
                current_block.txt_attn.proj.bias, next_block.txt_attn.proj.bias = next_block.txt_attn.proj.bias, current_block.txt_attn.proj.bias

            if shuffle_setting_img_mlp0 == "Shuffle" and idx in img_mlp0_layers_to_shuffle:
                current_block.img_mlp[0].weight, next_block.img_mlp[0].weight = next_block.img_mlp[0].weight, current_block.img_mlp[0].weight
                current_block.img_mlp[0].bias, next_block.img_mlp[0].bias = next_block.img_mlp[0].bias, current_block.img_mlp[0].bias

            if shuffle_setting_img_mlp2 == "Shuffle" and idx in img_mlp2_layers_to_shuffle:
                current_block.img_mlp[2].weight, next_block.img_mlp[2].weight = next_block.img_mlp[2].weight, current_block.img_mlp[2].weight
                current_block.img_mlp[2].bias, next_block.img_mlp[2].bias = next_block.img_mlp[2].bias, current_block.img_mlp[2].bias

            if shuffle_setting_txt_mlp0 == "Shuffle" and idx in txt_mlp0_layers_to_shuffle:
                current_block.txt_mlp[0].weight, next_block.txt_mlp[0].weight = next_block.txt_mlp[0].weight, current_block.txt_mlp[0].weight
                current_block.txt_mlp[0].bias, next_block.txt_mlp[0].bias = next_block.txt_mlp[0].bias, current_block.txt_mlp[0].bias

            if shuffle_setting_txt_mlp2 == "Shuffle" and idx in txt_mlp2_layers_to_shuffle:
                current_block.txt_mlp[2].weight, next_block.txt_mlp[2].weight = next_block.txt_mlp[2].weight, current_block.txt_mlp[2].weight
                current_block.txt_mlp[2].bias, next_block.txt_mlp[2].bias = next_block.txt_mlp[2].bias, current_block.txt_mlp[2].bias

            if shuffle_setting_img_mod == "Shuffle" and idx in img_mod_layers_to_shuffle:
                current_block.img_mod.lin, next_block.img_mod.lin = next_block.img_mod.lin, current_block.img_mod.lin

            if shuffle_setting_txt_mod == "Shuffle" and idx in txt_mod_layers_to_shuffle:
                current_block.txt_mod.lin, next_block.txt_mod.lin = next_block.txt_mod.lin, current_block.txt_mod.lin

            if shuffle_setting_imgnorm1 == "Identity" and idx in all_identity_to_shuffle:
                current_block.img_norm1 = torch.nn.Identity()

            if shuffle_setting_imgnorm1 == "Identity" and idx in all_identity_to_shuffle:
                current_block.img_norm2 = torch.nn.Identity()

            if shuffle_setting_imgnorm1 == "Identity" and idx in all_identity_to_shuffle:
                current_block.txt_norm1 = torch.nn.Identity()

            if shuffle_setting_imgnorm1 == "Identity" and idx in all_identity_to_shuffle:
                current_block.txt_norm2 = torch.nn.Identity()

        # Shuffle single block components based on the user's settings for MLP layers
        for idx in range(len(diffusion_model.single_blocks) - 2):  # Avoid going out of range
            current_block = diffusion_model.single_blocks[idx]
            next_block = diffusion_model.single_blocks[idx + 1]

            if shuffle_setting_single == "Skip" and idx in single_layers_to_shuffle:
                # Find the next target layer in the skip list that comes after the current one
                subsequent_layers_single = [i for i in single_layers_to_shuffle if i > idx]

                if subsequent_layers_single:
                    # Get the first subsequent layer after the current one to connect to
                    target_idx_single = min(subsequent_layers_single)
                    target_block_single = diffusion_model.single_blocks[target_idx_single]                   
                    # Bypass intermediate layers: directly connect the current layer's output to the target layer's input
                    current_block.forward = lambda *args, **kwargs: target_block_single.forward(*args, **kwargs)

                else:
                    # If no subsequent layers to skip to, just proceed as normal
                    print(f"No subsequent layers to skip after layer {idx}. Proceeding normally.")

            if shuffle_setting_lin1 == "Shuffle" and idx in lin1_layers_to_shuffle:
                current_block.linear1.weight, next_block.linear1.weight = next_block.linear1.weight, current_block.linear1.weight
                current_block.linear1.bias, next_block.linear1.bias = next_block.linear1.bias, current_block.linear1.bias
            
            if shuffle_setting_lin2 == "Shuffle" and idx in lin2_layers_to_shuffle:
                current_block.linear2.weight, next_block.linear2.weight = next_block.linear2.weight, current_block.linear2.weight
                current_block.linear2.bias, next_block.linear2.bias = next_block.linear2.bias, current_block.linear2.bias

            if shuffle_setting_prenorm == "Shuffle" and idx in prenorm_layers_to_shuffle:
                current_block.pre_norm.weight, next_block.pre_norm.weight = next_block.pre_norm.weight, current_block.pre_norm.weight
                current_block.pre_norm.bias, next_block.pre_norm.bias = next_block.pre_norm.bias, current_block.pre_norm.bias

            if shuffle_setting_sinmod == "Shuffle" and idx in sinmod_layers_to_shuffle:
                current_block.modulation.lin.weight, next_block.modulation.lin.weight = next_block.modulation.lin.weight, current_block.modulation.lin.weight
                current_block.modulation.lin.bias, next_block.modulation.lin.bias = next_block.modulation.lin.bias, current_block.modulation.lin.bias


        wrapped_model = comfy.model_patcher.ModelPatcher(model, load_device, offload_device)

        return (wrapped_model,)