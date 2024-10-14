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

class ConfusionCLIPLoader:
    def __init__(self):
        self.clip = None
        self.clip_name1 = None
        self.clip_name2 = None
        self.clip_type = None
        pass
        
    @classmethod
    def load_clip(self, clip_name1, clip_name2, clip_type):
        # Generate file paths for the checkpoint files
        clip_path1 = folder_paths.get_full_path_or_raise("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("clip", clip_name2)

        # Map the type to the appropriate CLIPType
        if clip_type == "sdxl":
            clip_type_enum = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif clip_type == "sd3":
            clip_type_enum = comfy.sd.CLIPType.SD3
        elif clip_type == "flux":
            clip_type_enum = comfy.sd.CLIPType.FLUX

        # Load model weights from file
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type_enum
        )
        return clip

class CLIPshuffleBase:
    def __init__(self, load_from_file=False):
        super().__init__()
        self.load_from_file = load_from_file
        self.clip_l = None
        self.t5xxl = None
        self.original_clip_l = None
        self.original_t5xxl = None
        self.clip_loader = ConfusionCLIPLoader()
           
    def get_original_models(self, clip_name1, clip_name2, model_type):
        if self.load_from_file:
            if self.original_clip_l is not None and self.original_t5xxl is not None:
                del self.original_clip_l
                del self.original_t5xxl
                self.original_clip_l = None
                self.original_t5xxl = None
                torch.cuda.empty_cache()
                print(f"CLIP + T5 Shuffle: Switching to load-file-always, deleted deepcopy at: {time.time()}")  

            if self.clip_l is not None and self.t5xxl is not None:
                del self.clip_l
                del self.t5xxl
                self.clip_l = None
                self.t5xxl = None
                torch.cuda.empty_cache()
                print(f"CLIP + T5 Shuffle: File-load-always is enabled, deleted models at: {time.time()}")  
                
            # Use the loader to fetch the models from file
            clip = self.clip_loader.load_clip(clip_name1, clip_name2, model_type)
            clip_l = clip.cond_stage_model.clip_l
            t5xxl = clip.cond_stage_model.t5xxl
            self.original_clip_l = clip_l
            self.original_t5xxl = t5xxl
            print(f"CLIP + T5 Shuffle: Loaded CLIP and T5 models from file at: {time.time()} (file-load-always)")
        else:
            if self.clip_l is None and self.t5xxl is None:
                clip = self.clip_loader.load_clip(clip_name1, clip_name2, model_type)
                clip_l = clip.cond_stage_model.clip_l
                t5xxl = clip.cond_stage_model.t5xxl
                self.original_clip_l = copy.deepcopy(clip_l)
                self.original_t5xxl = copy.deepcopy(t5xxl)
                print(f"CLIP + T5 Shuffle: Loaded CLIP and T5 models from file at: {time.time()} (file-load-once)")

            if self.original_clip_l is not None and self.original_t5xxl is not None:
                del self.original_clip_l
                del self.original_t5xxl
                torch.cuda.empty_cache()
                self.original_clip_l = copy.deepcopy(clip_l)
                self.original_t5xxl = copy.deepcopy(t5xxl)
                print(f"CLIP + T5 Shuffle: Original CLIP and T5 models deepcopy saved at: {time.time()}")

        clip_l_copy = self.original_clip_l
        t5xxl_copy = self.original_t5xxl

        return clip, clip_l_copy, t5xxl_copy
        
        
class CLIPshuffleLayersNode(CLIPshuffleBase):
    def __init__(self, load_from_file=False):
        load_from_file=load_from_file
        self.clip = None 
        self.clip_l = None
        self.t5xxl = None
        self.original_clip_l = None
        self.original_t5xxl = None
        self.clip_loader = ConfusionCLIPLoader()
        pass

    @classmethod
    def IS_CHANGED(c, **kwargs):
        sillytimestamp = time.time()
        return sillytimestamp

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("clip"),),
                "clip_name2": (folder_paths.get_filename_list("clip"),),
                "model_type": (["sdxl", "sd3", "flux"], {"default": "flux"}),
                "load_from_file": (["False", "True"], {"default": "True"}),
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
    CATEGORY = "zer0int/CLIP-Flux-Shuffle"

    def shuffle_clip_t5(self, clip_name1, clip_name2, model_type, load_from_file, shuffle_setting_clip_attn, clip_attn_layers, shuffle_setting_clip_mlp, clip_mlp_layers, shuffle_setting_t5_attn, t5_attn_layers, shuffle_setting_t5_mlp, t5_mlp_layers):
        # Determine if we are loading from file or from memory
        self.load_from_file = load_from_file == "True"
        
        # Fetch the CLIP and T5 models
        clip, clip_l, t5_model = self.get_original_models(clip_name1, clip_name2, model_type)

        
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

        timestamp = time.time() 
        
        # Re-wrap the modified models back into the FluxClipModel_ structure
        #new_clip = comfy.sd.create_clip(clip_l=clip_l, t5xxl=t5_model)
        clip.cond_stage_model.clip_l = clip_l
        clip.cond_stage_model.t5xxl = t5_model

        return (clip,)

class ConfusionUNETLoader:
    def __init__(self):
        self.model = None
        self.original_model = None
        self.unet_name = None
        self.weight_dtype = None
        pass
        
    @classmethod
    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return model


class FluxShuffleBase:
    def __init__(self, load_from_file=False):
        super().__init__()
        self.load_from_file = load_from_file
        self.model = None
        self.original_model= None
        self.unet_name = None
        self.weight_dtype = None
        self.unet_loader = ConfusionUNETLoader()
           
    def get_original_model(self, unet_name, weight_dtype):
        if self.load_from_file:
            if self.original_model is not None:
                del self.original_model
                self.original_model = None
                torch.cuda.empty_cache()
                print(f"Flux Shuffle: Switching to load-file-always, deleted deepcopy at: {time.time()}")  

            if self.model is not None:
                del self.model
                self.model = None
                torch.cuda.empty_cache()
                print(f"Flux Shuffle: File-load-always is enabled, deleted models at: {time.time()}")  
                
            # Use the loader to fetch the model from file
            model = self.unet_loader.load_unet(unet_name, weight_dtype)
            self.original_model = model
            if hasattr(self.original_model, 'model'):
                self.original_model = self.original_model.model
            print(f"Flux Shuffle: Loaded Flux from file at: {time.time()} (file-load-always)")
        else:
            if self.model is None:
                model = self.unet_loader.load_unet(unet_name, weight_dtype)
                self.original_model = copy.deepcopy(model)
                if hasattr(self.original_model, 'model'):
                    self.original_model = self.original_model.model
                print(f"Flux Shuffle: Loaded Flux from file at: {time.time()} (file-load-once)")

            if self.original_model is not None:
                del self.original_model
                torch.cuda.empty_cache()
                self.original_model = copy.deepcopy(model)
                if hasattr(self.original_model, 'model'):
                    self.original_model = self.original_model.model
                print(f"Flux Shuffle: Flux model deepcopy saved at: {time.time()}")

        modelflux = self.original_model

        return modelflux

class ShuffleFluxLayersNode(FluxShuffleBase):
    def __init__(self, load_from_file=False):
        self.load_from_file = load_from_file
        self.model = None
        self.original_model = None
        self.unet_name = None
        self.weight_dtype = None
        self.unet_loader = ConfusionUNETLoader()
        pass

    @classmethod
    def IS_CHANGED(c, **kwargs):
        sillytimestamp = time.time()
        return sillytimestamp

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e4m3fn_fast"], {"default": "default"}),
                "load_from_file": (["False", "True"], {"default": "True"}),            
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
    CATEGORY = "zer0int/CLIP-Flux-Shuffle"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "shuffle_and_apply"
   
    def shuffle_and_apply(self, unet_name, weight_dtype, load_from_file, shuffle_setting_double, double_layers, shuffle_setting_attn_img, attn_layers_img, shuffle_setting_attn_txt, attn_layers_txt, shuffle_setting_attn_txt_proj, attn_layers_txtproj, shuffle_setting_img_mlp0, img_mlp0_layers, shuffle_setting_img_mlp2, img_mlp2_layers, shuffle_setting_img_mod, img_mod_layers, shuffle_setting_txt_mod, txt_mod_layers, shuffle_setting_single, single_layers, shuffle_setting_lin1, lin1_layers, shuffle_setting_lin2, lin2_layers, shuffle_setting_prenorm, prenorm_layers, shuffle_setting_sinmod, sinmod_layers, shuffle_setting_txt_mlp0, txt_mlp0_layers, shuffle_setting_txt_mlp2, txt_mlp2_layers, shuffle_setting_txtnorm1, shuffle_setting_txtnorm2, all_identity, shuffle_setting_imgnorm1, shuffle_setting_imgnorm2):
        self.load_from_file = load_from_file == "True"
        
        # Fetch Flux model
        model = self.get_original_model(unet_name, weight_dtype)       
        diffusion_model = model.diffusion_model
        
        offload_device = mm.unet_offload_device()
        load_device = get_torch_device()     
        
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

        timestamp = time.time()
        wrapped_model = comfy.model_patcher.ModelPatcher(model, load_device, offload_device)

        return (wrapped_model,)
