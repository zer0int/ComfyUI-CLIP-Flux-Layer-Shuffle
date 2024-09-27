# Nodes for messing with a model's layers in ComfyUI.
## For CLIP, T5, and Flux! ❗🤖🔀🤖❓

- CLI script (Diffusers) that allows you to shuffle layer components (attention, MLP, ...) around.
- Agonizing CPU offload, minutes to image - but runs on 4 GB VRAM. Insert your own Flux pipe to optimize it.
- Check out the example-images and corresponding settings txt to get started!
- ComfyUI nodes: Put the folder "ComfyUI_CLIPFluxShuffle" into "ComfyUI/custom_nodes". Launch Comfy.
- Right click -> Add Node -> CLIP-Flux-Shuffle. Or use workflows from 'workflows' folder.
- You can use the CLIP + T5 nodes to see what each AI contributes (see "hierarchical" image for an idea)!
- You probably can't use the Flux node. :/ It peaks at ~70 GB RAM use. Yes, in ADDITION to OS stuff.
- TO-DO: Need to figure out memory / model management and find a better way.
- I am currently creating a deepcopy of a deepcopy (of the full model), as it appears buffers etc also get messed up during layer shuffle. Restoring from state_dict (deepcopy thereof) didn't seem to work. Re-loading / instantiating the model from file every time is also pretty unfeasible. Pull requests or even simple hint drops welcome (please open an issue)! Thank you!
--------
![1-mega-node](https://github.com/user-attachments/assets/29cd2edb-9b87-41ac-ae28-dcbc30cb25eb)

![2-cli-settings-available](https://github.com/user-attachments/assets/fd195761-cfca-4e62-acba-ed22b638197f)
