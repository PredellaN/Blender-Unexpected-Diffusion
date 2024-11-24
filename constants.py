from collections import namedtuple

DiffModel = namedtuple('SDModel', ['id', 'label', 'type'])

DIFFUSION_MODELS = [
    DiffModel('stablediffusionapi/NightVision_XL', 'NightVision_XL', 'SDXL'),
    DiffModel('stablediffusionapi/nightvisionxl', 'NightVision_XL 0.9.0', 'SDXL'),
    
    DiffModel('SG161222/RealVisXL_V4.0', 'RealVisXL V4.0', 'SDXL'),

    DiffModel('stabilityai/stable-diffusion-xl-base-1.0', 'SDXL Base', 'SDXL'),
    DiffModel('stabilityai/sdxl-turbo', 'SDXL Turbo', 'SDXL'),
    DiffModel('Vargol/sdxl-lightning-4-steps', 'SDXL-Lightning', 'SDXL'),

    DiffModel('segmind/SSD-1B', 'SSD-1B', 'SDXL'),

    DiffModel('Lykon/dreamshaper-xl-1-0', 'Dreamshaper XL 1.0', 'SDXL'),
    DiffModel('Lykon/dreamshaper-xl-turbo', 'Dreamshaper XL Turbo', 'SDXL'),

    DiffModel('stablediffusionapi/juggernaut-xl-v7', 'Juggernaut XL v7', 'SDXL'),
    DiffModel('RunDiffusion/Juggernaut-X-Hyper', 'Juggernaut-X-Hyper', 'SDXL'),

    DiffModel('playgroundai/playground-v2-1024px-aesthetic', 'Playground V2 Aesthetic', 'SDXL'),
    DiffModel('playgroundai/playground-v2.5-1024px-aesthetic', 'Playground V2.5 Aesthetic', 'SDXL'),

    DiffModel('black-forest-labs/FLUX.1-schnell', 'FLUX.1-schnell', 'FLUX'),

    DiffModel('stabilityai/stable-diffusion-3.5-medium', 'Stable Diffusion 3.5 Medium', 'SD3'),
    DiffModel('stabilityai/stable-diffusion-3.5-large', 'Stable Diffusion 3.5 Large', 'SD3'),
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
    'ValouF-pimento/ControlNet_SDXL_tile_upscale': {'name': 'ControlNet_SDXL_tile_upscale', 'model_type': 'diffusers'},
    'TheMistoAI/MistoLine': {'name': 'MistoLine', 'model_type': 'diffusers'},
    'xinsir/controlnet-tile-sdxl-1.0': {'name': 'controlnet-tile-sdxl-1.0', 'model_type': 'diffusers'},
    'xinsir/controlnet-union-sdxl-1.0' : {'name': 'controlnet-union-sdxl-1.0', 'model_type': 'diffusers'},
}

T2I_MODELS = {
    'TencentARC/t2i-adapter-canny-sdxl-1.0': {'name': 't2i-adapter-canny-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-sketch-sdxl-1.0': {'name': 't2i-adapter-sketch-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-lineart-sdxl-1.0': {'name': 't2i-adapter-lineart-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-depth-midas-sdxl-1.0': {'name': 't2i-adapter-depth-midas-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-depth-zoe-sdxl-1.0': {'name': 't2i-adapter-depth-zoe-sdxl-1.0', 'model_type': 'diffusers'},
    'TencentARC/t2i-adapter-openpose-sdxl-1.0': {'name': 't2i-adapter-openpose-sdxl-1.0', 'model_type': 'diffusers'},
}