pipeline_settings = {
    "StableDiffusionXLControlNetInpaintPipeline": ['negative_prompt', "controlnet_model", "mask_image", "init_image", ],
    "StableDiffusionXLControlNetImg2ImgPipeline": ['negative_prompt', "controlnet_model", "init_image", ],
    "StableDiffusionXLControlNetPipeline": ['negative_prompt', "controlnet_model", ],
    "StableDiffusionXLAdapterPipeline": ['negative_prompt', "t2i_model" ],
    "StableDiffusionXLInpaintPipeline": ['negative_prompt', "image", "mask_image", ],
    "StableDiffusionXLImg2ImgPipeline": ['negative_prompt', "image", "strength"],
    "StableDiffusionXLPipeline": ['negative_prompt', ],
    "FluxPipeline": [],
}

for key in pipeline_settings:
    pipeline_settings[key] = ["prompt"] + ["width"] + ["height"] + ["generator"] + ["num_inference_steps"] + ["guidance_scale"] + pipeline_settings[key]