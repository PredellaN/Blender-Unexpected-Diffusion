SD_MODELS = [
    ('stablediffusionapi/NightVision_XL','NightVision_XL', ""),
    ('stabilityai/stable-diffusion-xl-base-1.0','SDXL Base', ""),
    ('segmind/SSD-1B','SSD-1B', ""),
    ('stabilityai/sdxl-turbo','SDXL Turbo', ""),
    ('Lykon/dreamshaper-xl-turbo','Dreamshaper XL Turbo',''),
    ('Lykon/dreamshaper-xl-1-0','Dreamshaper XL 1.0',''),
    ('stablediffusionapi/juggernaut-xl-v7','Juggernaut XL v7',''),
    ('playgroundai/playground-v2-1024px-aesthetic','Playground V2 Aesthetic', ""),
]

CONTROLNET_MODELS = {
    'diffusers/controlnet-depth-sdxl-1.0': {'name': 'controlnet-depth-sdxl-1.0', 'model_type': 'diffusers'},
    'diffusers/controlnet-depth-sdxl-1.0-small': {'name': 'controlnet-depth-sdxl-1.0-small', 'model_type': 'diffusers'},
    'diffusers/controlnet-depth-sdxl-1.0-mid': {'name': 'controlnet-depth-sdxl-1.0-mid', 'model_type': 'diffusers'},
    'diffusers/controlnet-canny-sdxl-1.0': {'name': 'controlnet-canny-sdxl-1.0', 'model_type': 'diffusers'},
    'diffusers/controlnet-canny-sdxl-1.0-small': {'name': 'controlnet-canny-sdxl-1.0-small', 'model_type': 'diffusers'},
    'diffusers/controlnet-canny-sdxl-1.0-mid': {'name': 'controlnet-canny-sdxl-1.0-mid', 'model_type': 'diffusers'},
    'Nacholmo/controlnet-qr-pattern-sdxl': {'name': 'controlnet-qr-pattern-sdxl', 'model_type': 'diffusers'},
    'SargeZT/sdxl-controlnet-seg': {'name': 'sdxl-controlnet-seg', 'model_type': 'diffusers'},
    'SargeZT/controlnet-sd-xl-1.0-softedge-dexined': {'name': 'controlnet-sd-xl-1.0-softedge-dexined', 'model_type': 'diffusers'},
    'monster-labs/control_v1p_sdxl_qrcode_monster': {'name': 'control_v1p_sdxl_qrcode_monster', 'model_type': 'diffusers'},
    'diffusers/controlnet-zoe-depth-sdxl-1.0': {'name': 'controlnet-zoe-depth-sdxl-1.0', 'model_type': 'diffusers'},
}

T2I_MODELS = {
    'TencentARC/t2i-adapter-canny-sdxl-1.0': {'name': 't2i-adapter-canny-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-sketch-sdxl-1.0': {'name': 't2i-adapter-sketch-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-lineart-sdxl-1.0': {'name': 't2i-adapter-lineart-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-depth-midas-sdxl-1.0': {'name': 't2i-adapter-depth-midas-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-depth-zoe-sdxl-1.0': {'name': 't2i-adapter-depth-zoe-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-openpose-sdxl-1.0': {'name': 't2i-adapter-openpose-sdxl-1.0', 'model_type': 'diffusers'},
}