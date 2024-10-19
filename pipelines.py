pipeline_settings = {
    "StableDiffusionXLPipeline": ['negative_prompt'],

    "StableDiffusionXLImg2ImgPipeline": ['negative_prompt', "image", "strength"],
    "StableDiffusionXLInpaintPipeline": ['negative_prompt', "image", "mask_image", "strength"],

    "StableDiffusionXLControlNetInpaintPipeline": ['negative_prompt', "image", "mask_image", "strength", "controlnet_model", "controlnet_conditioning_scale", "control_image"],
    "StableDiffusionXLControlNetImg2ImgPipeline": ['negative_prompt', "image", "strength", "controlnet_model", "controlnet_conditioning_scale", "control_image"],
    "StableDiffusionXLControlNetPipeline": ['negative_prompt', "image", "controlnet_model", "controlnet_conditioning_scale"],

    "StableDiffusionXLAdapterPipeline": ['negative_prompt', "image", "t2i_model", "adapter_conditioning_scale"],

    "FluxPipeline": [],
    "FluxImg2ImgPipeline": ["image", "strength"],
}

for key in pipeline_settings:
    pipeline_settings[key] = ["prompt"] + ["width"] + ["height"] + ["generator"] + ["num_inference_steps"] + ["guidance_scale"] + pipeline_settings[key]