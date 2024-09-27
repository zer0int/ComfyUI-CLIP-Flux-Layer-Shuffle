import os
import torch
from diffusers import FluxPipeline
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from transformers import T5EncoderModel, T5Config, AutoTokenizer
import warnings
# Suppress warnings like torch future warning spam
warnings.simplefilter("ignore")

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== User-defined Variables ==========

# Memory settings
torch_dtype = torch.bfloat16
# Enable low VRAM mode (set one to True, the other to False)
enable_cpu_offload = False # tends to be just over 24 GB unless no GUI (monitor) used. :(
enable_sequential_cpu_offload = True # Super slow, but <4 GB VRAM & 30 GB RAM instead

local_files_only = False # True for offline / Set to False if need to download the models. 
# Set True if have unstable internet (or just a VPN) connection RANDOMLY giving you a 
# AttributeError: 'FluxPipeline' object has no attribute 'device'.

# Choose CLIP model and prompt length
# norm, long: my fine-tunes of CLIP and CLIP-L. orgL, oai = pre-trained / original.
clipmodel = 'norm'  # Options: 'norm', 'long', 'oai', 'orgL'
selectedprompt = 'short'  # Options: 'tiny', 'short', 'med', 'long'

# Set seed for deterministic / reproducible results
seed = 4255330358390669
guidance_scale=3.5
num_inference_steps=20

imagename = "cat" # for filename

# Attn: Attention, confuse what AI is looking for. More subtle and fun. Try 2,3,4 with the Text Encoders, esp. T5!
# MLP: Multi-Layer perceptron. The features that make meaning based on attention. Easily destructive.
# Layer: Shuffles whole Layer (Attn + MLP that belongs with it). Model takes it much better, but confusion ensues.
# Ident: Do it on a few Layers, and you can see what happens without this model. So fatal, it uses Flux to generate a noise pattern.
# F, FF: FeedForward Layers. Shuffle them, and the model can't carry over Layers to make meaning. Also destructive.

# Specify shuffle settings
shuffle_setting_clip = "MLP"  # Options: "None", "MLP", "Layer", "Attn", "Ident"
shuffle_setting_t5 = "MLP"     # Options: "None", "MLP", "Layer", "Attn", "Ident"
shuffle_setting_flux_single = "MLP"  # Options: "None", "MLP", "Layer", "Ident"
shuffle_setting_flux_double = "Attn"  # Options: "None", "Layer", "Attn", "AttnAdd", "Ident", "F", "FF"

# Specify layers to shuffle for each model, if not 'None'
layer_range_clip = [2, 3, 4] # 0-11
layer_range_t5 = [2, 3, 4] # 1-23
layer_range_flux_single = [13, 14] # 0-37
layer_range_flux_double = [5, 6, 7] # 0-18

# ========== Model Setup ==========

# Set model configuration based on clipmodel choice
if clipmodel == "long":
    model_id = "zer0int/LongCLIP-GmP-ViT-L-14"
    maxtokens = 248
elif clipmodel == "orgL":
    model_id = "zer0int/LongCLIP-L-Diffusers"
    maxtokens = 248
elif clipmodel == "norm":
    model_id = "zer0int/CLIP-GmP-ViT-L-14"
    maxtokens = 77
else:  # clipmodel == "oai"
    model_id = "openai/clip-vit-large-patch14"
    maxtokens = 77

# Set prompt based on selectedprompt
prompts = {
    "long": "A photo of a fluffy Maine Coon cat, majestic in size, with long, luxurious fur of silver and charcoal tones. The cat has striking heterochromatic eyes, one a deep sapphire blue, the other a brilliant emerald green, giving it an aura of mystery and intrigue. The Maine Coon is wearing a tiny, playful hat tilted slightly to the side, a miniature top hat made of soft velvet, in a deep shade of royal purple, with a golden ribbon tied around the base. The cat is wearing a delicate silver chain necklace with a small pendant in the shape of a crescent moon. The cat is sitting up proudly, holding a large wooden sign between its front paws. The sign is made from old, weathered wood. Written on the sign in elegant, hand-painted script are the words 'Long CLIP is long, Long CAT is lovely!'. The cat is sitting on a lush green patch of grass, with small wildflowers blooming around it on a sunny day with some cumulus clouds in the sky.",
    "med": "A photo of a fluffy Maine Coon cat with long fur of silver and charcoal tones. The cat has heterochromatic eyes, one eye is a deep sapphire blue, the other eye is a brilliant emerald green. It is wearing a playful miniature top hat made of soft velvet in a deep shade of royal purple. The cat is holding a large wooden sign made from old, weathered wood between its front paws. Written on the sign in hand-painted script are the words 'Long CLIP is long, Long CAT is lovely!'. Background lush green patch of grass with small wildflowers.",
    "short": "A photo of a Maine Coon cat with heterochromatic eyes, one eye is sapphire blue, the other eye is emerald green. It is wearing a miniature top hat. the cat is holding a sign made from weathered wood. Written on the sign in hand-painted script are the words 'long CLIP is long and long CAT is lovely!'. Background grass with wildflowers.",
    "tiny": "A photo of a Maine Coon with heterochromatic eyes, one eye is sapphire blue, the other eye is emerald green. The cat is holding a wooden sign. The sign says 'long CLIP is long and long CAT is lovely!'. Background meadow.",
}

prompt = prompts[selectedprompt]

# Load CLIP model and processor
config = CLIPConfig.from_pretrained(model_id)
config.text_config.max_position_embeddings = maxtokens

clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch_dtype, config=config, local_files_only=local_files_only).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=maxtokens, return_tensors="pt", truncation=True, local_files_only=local_files_only)

# Load FluxPipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch_dtype, local_files_only=local_files_only)

# Set CLIP tokenizer and text encoder in pipeline to use custom model
pipe.tokenizer = clip_processor.tokenizer
pipe.text_encoder = clip_model.text_model
pipe.tokenizer_max_length = maxtokens
pipe.text_encoder.dtype = torch_dtype

# ========== Shuffle Components ==========

def shuffle_components_clip(pipe, layer_range, shuffle_setting):
    for layer_idx in layer_range_clip:
        layer1 = pipe.text_encoder.encoder.layers[layer_idx]
        layer2 = pipe.text_encoder.encoder.layers[layer_idx + 1]

        if shuffle_setting_clip == "MLP":
            # Shuffle only the MLP components
            layer1.mlp.fc1.weight, layer2.mlp.fc2.weight = layer2.mlp.fc1.weight, layer1.mlp.fc2.weight
            layer1.mlp.fc1.bias, layer2.mlp.fc2.bias = layer2.mlp.fc1.bias, layer1.mlp.fc2.bias

        elif shuffle_setting_clip == "Attn":
            # Shuffle only the attention components
            layer1.self_attn.out_proj.weight, layer2.self_attn.out_proj.weight = layer2.self_attn.out_proj.weight, layer1.self_attn.out_proj.weight
            layer1.self_attn.out_proj.bias, layer2.self_attn.out_proj.bias = layer2.self_attn.out_proj.bias, layer1.self_attn.out_proj.bias

        elif shuffle_setting_clip == "Layer":
            # Shuffle the whole layer (everything in the layer)
            layer1, layer2 = layer2, layer1

        elif shuffle_setting_clip == "Ident":
            # Set the LayerNorm to be an identity operation (no-op)
            layer1.layer_norm1 = torch.nn.Identity()
            layer2.layer_norm2 = torch.nn.Identity()

        elif shuffle_setting_clip == "None":
            # Do nothing
            pass

def shuffle_components_t5(pipe, layer_range, shuffle_setting):
    for layer_idx in layer_range_t5:
        layer1 = pipe.text_encoder_2.encoder.block[layer_idx]
        layer2 = pipe.text_encoder_2.encoder.block[layer_idx + 1]

        if shuffle_setting_t5 == "MLP":
            # Shuffle only the MLP components
            layer1.layer[1].DenseReluDense.wi_0.weight, layer2.layer[1].DenseReluDense.wi_1.weight = layer2.layer[1].DenseReluDense.wi_0.weight, layer1.layer[1].DenseReluDense.wi_1.weight
            layer1.layer[1].DenseReluDense.wi_0.bias, layer2.layer[1].DenseReluDense.wi_1.bias = layer2.layer[1].DenseReluDense.wi_0.bias, layer1.layer[1].DenseReluDense.wi_1.bias

        elif shuffle_setting_t5 == "Attn":
            # Shuffle only the attention components
            layer1.layer[0].SelfAttention.weight, layer2.layer[1].SelfAttention.weight = layer2.layer[1].SelfAttention.weight, layer1.layer[0].SelfAttention.weight
            layer1.layer[0].SelfAttention.bias, layer2.layer[1].SelfAttention.bias = layer2.layer[1].SelfAttention.bias, layer1.layer[0].SelfAttention.bias

        elif shuffle_setting_t5 == "Layer":
            # Shuffle the whole layer (everything in the layer)
            layer1.layer, layer2.layer = layer2.layer, layer1.layer

        elif shuffle_setting_t5 == "Ident":
            # Set the LayerNorm to be an identity operation (no-op)
            layer1.layer[0].layer_norm = torch.nn.Identity()
            layer2.layer[1].layer_norm = torch.nn.Identity()

        elif shuffle_setting_t5 == "None":
            # Do nothing
            pass

def shuffle_components_flux_single(pipe, layer_range_flux_single, shuffle_setting_flux_single):
    for layer_idx in layer_range_flux_single:
        layer1 = pipe.transformer.single_transformer_blocks[layer_idx]
        layer2 = pipe.transformer.single_transformer_blocks[layer_idx + 1]

        if shuffle_setting_flux_single == "MLP":
            # Shuffle only the MLP components
            layer1.proj_mlp.weight, layer2.proj_out.weight = layer2.proj_mlp.weight, layer1.proj_out.weight
            layer1.proj_mlp.bias, layer2.proj_out.bias = layer2.proj_mlp.bias, layer1.proj_out.bias          
     
        elif shuffle_setting_flux_single == "Layer":
            # Shuffle the whole layer (everything in the layer)
            layer1, layer2 = layer2, layer1

        elif shuffle_setting_flux_single == "Ident":
            # Set the LayerNorm to be an identity operation (no-op)
            layer1.norm.norm = torch.nn.Identity()

        elif shuffle_setting_flux_single == "None":
            # Do nothing
            pass
            
def shuffle_components_flux_double(pipe, layer_range_flux_double, shuffle_setting_flux_double):
    for layer_idx in layer_range_flux_double:
        layer1 = pipe.transformer.transformer_blocks[layer_idx]
        layer2 = pipe.transformer.transformer_blocks[layer_idx + 1]

        if  shuffle_setting_flux_double == "Attn":
            # Shuffle only the attention components
            layer1.attn.to_out[0].weight, layer2.attn.to_out[0].weight = layer2.attn.to_out[0].weight, layer1.attn.to_out[0].weight
            layer1.attn.to_out[0].bias, layer2.attn.to_out[0].bias = layer2.attn.to_out[0].bias, layer1.attn.to_out[0].bias

        if  shuffle_setting_flux_double == "AttnAdd":
            # Shuffle only the attention components
            layer1.attn.to_add_out.weight, layer2.attn.to_add_out.weight = layer2.attn.to_add_out.weight, layer1.attn.to_add_out.weight
            layer1.attn.to_add_out.bias, layer2.attn.to_add_out.bias = layer2.attn.to_add_out.bias, layer1.attn.to_add_out.bias

        elif shuffle_setting_flux_double == "Layer":
            # Shuffle the whole layer (everything in the layer)
            layer1, layer2 = layer2, layer1

        elif shuffle_setting_flux_double == "Ident":
            # Set the LayerNorm to be an identity operation (no-op)
            layer1.norm2 = torch.nn.Identity()
            layer1.norm2_context = torch.nn.Identity()
            layer2.norm2 = torch.nn.Identity()
            layer2.norm2_context = torch.nn.Identity()

        elif shuffle_setting_flux_double == "F":
            # Set the LayerNorm to be an identity operation (no-op)
            layer1.ff.net[0].proj, layer2.ff.net[0].proj = layer2.ff.net[0].proj, layer1.ff.net[0].proj

        elif shuffle_setting_flux_double == "FF":
            # Set the LayerNorm to be an identity operation (no-op)
            layer1.ff.net[0].proj, layer2.ff.net[0].proj = layer2.ff.net[0].proj, layer1.ff.net[0].proj
            layer1.ff_context.net[0].proj, layer2.ff_context.net[0].proj = layer2.ff_context.net[0].proj, layer1.ff_context.net[0].proj


        elif shuffle_setting_flux_double == "None":
            # Do nothing
            pass

# Shuffle the components based on user-defined settings
shuffle_components_clip(pipe, layer_range_clip, shuffle_setting_clip)
shuffle_components_t5(pipe, layer_range_t5, shuffle_setting_t5)
shuffle_components_flux_single(pipe, layer_range_flux_single, shuffle_setting_flux_single)
shuffle_components_flux_double(pipe, layer_range_flux_double, shuffle_setting_flux_double)

# ========== Pipeline Execution ==========

# Tokenization and check for token count
tokens = clip_processor([prompt], padding="max_length", max_length=maxtokens, return_tensors="pt", truncation=True)
non_padding_count = torch.sum(tokens['input_ids'][0] != clip_processor.tokenizer.pad_token_id).item()

print(f"\nNumber of tokens: {tokens['input_ids'].shape[1]}")
print(f"Number of non-padding tokens: {non_padding_count}\n")

# Enable low VRAM modes if necessary
if enable_cpu_offload:
    pipe.enable_model_cpu_offload()
elif enable_sequential_cpu_offload:
    pipe.enable_sequential_cpu_offload()

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate the output image
generator = torch.manual_seed(seed)
out = pipe(
    prompt=prompt,
    guidance_scale=guidance_scale,
    height=1024,
    width=1024,
    num_inference_steps=num_inference_steps,
    generator=generator
).images[0]

out.save(f"{imagename}_M-{clipmodel}_P-{selectedprompt}_S-{shuffle_setting_clip}-{shuffle_setting_t5}-{shuffle_setting_flux_single}-{shuffle_setting_flux_double}.png")
