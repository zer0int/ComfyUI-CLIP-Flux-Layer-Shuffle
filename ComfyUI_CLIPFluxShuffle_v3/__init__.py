from . import clipfluxshuffle as clipfluxshuffle

NODE_CLASS_MAPPINGS = {
    "ShuffleFluxLayersNode": clipfluxshuffle.ShuffleFluxLayersNode,
    "CLIPshuffleLayersNode": clipfluxshuffle.CLIPshuffleLayersNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShuffleFluxLayersNode": "Shuffle Flux",
    "CLIPshuffleLayersNode": "Shuffle CLIP + T5",
}