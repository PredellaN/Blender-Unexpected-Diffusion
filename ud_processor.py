from diffusers import T2IAdapter, MultiAdapter, DPMSolverMultistepScheduler, StableCascadeCombinedPipeline, StableCascadePriorPipeline, StableCascadeDecoderPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline, StableDiffusionXLAdapterPipeline, StableDiffusionUpscalePipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL
from diffusers.pipelines.stable_cascade.modeling_stable_cascade_common import StableCascadeUnet

import bpy
import os
import numpy as np
import torch
from PIL import Image
from PIL import Image, ImageEnhance
from realesrgan_ncnn_py import Realesrgan
from . import gpudetector

from .constants import SD_MODELS, CONTROLNET_MODELS, T2I_MODELS

current_dir = os.path.dirname(os.path.realpath(__file__))

# Install opencv-python-headless instead of regular opencv-python! Or you'll run into xcb conflicts

def round_to_nearest(n):
    if n - int(n) < 0.5:
        return int(n)
    else:
        return int(n) + 1
    
def blender_image_to_pil(blender_image):
    # Ensure the image is not None
    if blender_image is None:
        raise ValueError("No Blender image provided")

    # Get the image data as a numpy array
    pixels = np.array(blender_image.pixels[:])  # Flatten pixel values
    size = blender_image.size[0], blender_image.size[1]  # Image dimensions

    # Reshape and convert the array to a suitable format
    pixels = np.reshape(pixels, (size[1], size[0], 4))  # Assuming RGBA
    pixels = np.flip(pixels, axis=0)  # Flip the image vertically
    pixels = (pixels * 255).astype(np.uint8)  # Convert to 8-bit per channel

    # Create and return a PIL image
    return Image.fromarray(pixels, 'RGBA')

def create_alpha_mask(image):

    if image.mode != 'RGBA':
        raise ValueError("Image does not have an alpha channel")

    alpha = image.split()[-1]
    inverted_alpha = Image.eval(alpha, lambda a: 255 - a)

    mask = inverted_alpha.convert('L')

    return mask

def is_mask_almost_black(mask, tolerance=5):

    if mask.mode != 'L':
        mask = mask.convert('L')

    mask_array = np.array(mask)
    avg_pixel = np.mean(mask_array)

    return avg_pixel < tolerance

class UD_Processor():
    prompt_adds = ", highly detailed, beautiful, 4K, photorealistic, high resolution"
    negative_prompt_adds = ", text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, bad, distortion, twisted, grainy, duplicate, error, pixelated, fake, glitch, overexposed, bad-contrast"
    refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"
    vae_model = "madebyollin/sdxl-vae-fp16-fix"

    upscale_strength = 0.35
    upscaling_rate = 2
    upscaling_steps = 10

    loaded_model = None
    loaded_model_type = None
    loaded_vae = None
    loaded_controlnets = None
    loaded_t2i = None

    ws = None

    def run(self, ws, params):
        self.ws = ws

        target_width = round((params['width'] * params['scale'] / 100) / 16) * 16
        target_height = round((params['height'] * params['scale'] / 100) / 16) * 16

        init_image = blender_image_to_pil(params['init_image_slot']).resize((target_width, target_height)) if params.get('init_image_slot') else None

        if params['init_mask_slot']:
            mask_image = blender_image_to_pil(params['init_mask_slot']).resize((target_width, target_height)).convert("RGB")
        else:
            if init_image:
                mask_image = create_alpha_mask(init_image)
                if is_mask_almost_black(mask_image):
                    mask_image = None
            else:
                mask_image = None

        pipeline_type = self.determine_pipeline_type(params, init_image, mask_image)
        
        if 'controlnet_image_slot' in params:
            controlnet_image = [blender_image_to_pil(slot).resize((target_width, target_height)).convert("RGB") for slot in params['controlnet_image_slot']]

        if 't2i_image_slot' in params:
            t2i_image = [blender_image_to_pil(slot).resize((target_width, target_height)).convert("RGB") for slot in params['t2i_image_slot']]
            
        overrides={
            'prompt': params['prompt'] + self.prompt_adds,
            'negative_prompt': params['negative_prompt'] + self.negative_prompt_adds,
            'width': target_width,
            'height': target_height,
            'generator': torch.manual_seed(params["seed"]),
        }

        params['steps_multiplier'] = 0.1 if 'turbo' in params['model'] else 1

        if 'controlnet_model' in params:
            overrides.update({
                'image': init_image if init_image else controlnet_image,
                'controlnet_conditioning_scale': params['controlnet_factor'],
            })
            if init_image:
                overrides.update({
                    'strength': params['denoise_strength'],
                    'num_inference_steps': round_to_nearest(params['inference_steps'] / params['denoise_strength']),
                    'control_image': controlnet_image,
                })

        elif 't2i_model' in params:
            if len(params['t2i_model']) > 1:
                overrides.update({
                    'image': t2i_image,
                    'adapter_conditioning_scale': params['t2i_factor'],
                })     
            else:
                overrides.update({
                    'image': t2i_image[0],
                    'adapter_conditioning_scale': params['t2i_factor'][0],
                })                

        else:
            overrides.update({
                'denoising_end': params['high_noise_frac']
                })
            if init_image:
                overrides.update({
                    'image': init_image.convert("RGB"),
                    'num_inference_steps': round_to_nearest(params['inference_steps'] / params['denoise_strength']),
                    'strength': params['denoise_strength'],
                })

        if init_image and mask_image:
            overrides.update({
                'mask_image': mask_image,
            })

        # if params['model'] in ['stabilityai/stable-cascade']:
        #     decoded_image = self.run_cascade_pipeline(
        #         params=params,
        #     )
        #     return decoded_image
        # else:
        latent_image = self.run_sdxl_pipeline(
            params=params,
            pipeline_type=pipeline_type,
            pipeline_model=params['model'],
            vae_model=self.vae_model,
            controlnet_models=params.get('controlnet_model', []),
            t2i_models=params.get('t2i_model', []),
            overrides=overrides
        )

        if latent_image is not None:
            if params['high_noise_frac'] < 1:
            # Start Refining
                overrides = {
                    'prompt': params['prompt'] + self.prompt_adds,
                    'negative_prompt': params['negative_prompt'] + ' hdr ' + self.negative_prompt_adds,
                    'image': latent_image,
                    'strength': params['refiner_strength'],
                }

                if 'controlnet_model' not in params:
                    overrides['denoising_start'] = params['high_noise_frac']
                    overrides['num_inference_steps'] = int (params['inference_steps'])
                else:
                    overrides['num_inference_steps'] =  round_to_nearest(params['inference_steps'] / params['refiner_strength'])

                decoded_image = self.run_sdxl_pipeline(
                    params=params,
                    pipeline_type='StableDiffusionXLImg2ImgPipeline',
                    pipeline_model=self.refiner_model,
                    vae_model=self.vae_model,
                    overrides=overrides,
                    output_type='pil'
                )
            else:
                with torch.no_grad():
                    image = self.pipe.vae.decode(latent_image / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                    decoded_image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
            
            return decoded_image
        else:
            return None

    def upscale(self, ws, params): 
        self.ws = ws

        image = Image.open(params['temp_image_filepath'])

        current_width = round_to_nearest(params['width']/16)*16
        current_height = round_to_nearest(params['height']/16)*16
        
        params['steps_multiplier'] = 0.5 if 'turbo' in params['model'] else 1

        if params['mode'] == 'upscale_re':
            
            self.ws.ud.progress = 0
            self.ws.ud.progress_text = 'Resizing with Realesrgan ...'

            # Resize to 4x using realesrgan
            realesrgan = Realesrgan(gpuid = gpudetector.get_nvidia_gpu(), model = 4)
            image = realesrgan.process_pil(image)
            realesrgan = None
            upscaled_image = image.resize((current_width * 2, current_height * 2), Image.Resampling.LANCZOS)
            contrast=1.1

        elif params['mode'] == 'upscale_sd':
            # Resize to 4x using stable-diffusion-x4-upscaler
            self.unload()   
            model_id = "stabilityai/stable-diffusion-x4-upscaler"
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipe = self.pipe.to("cuda")
            self.pipe.enable_attention_slicing()
            upscaled_image = self.pipe(
                    prompt=params['prompt'],
                    image=image.convert("RGB"),
                    noise_level=5,
                    num_inference_steps=25,
                ).images[0]
            upscaled_image = upscaled_image.resize((current_width * 2, current_height * 2), Image.Resampling.LANCZOS)
            contrast=1.1

        # Enhance the contrast by 10% as the upscale reduces the contrast
        enhancer = ImageEnhance.Contrast(upscaled_image)
        upscaled_image = enhancer.enhance(contrast)

        upscaled_image.save(params['temp_image_filepath'])

        # Refine upscaled image
        overrides = {
                'prompt': params['prompt'] + self.prompt_adds,
                'negative_prompt': params['negative_prompt'] + ' hdr ' + self.negative_prompt_adds,
                'image': upscaled_image.convert('RGB'),
                'strength': self.upscale_strength,
                'num_inference_steps': round_to_nearest(self.upscaling_steps / self.upscale_strength),
                'guidance_scale': 5,
            }

        for model in [params['model']]:
            decoded_image = self.run_sdxl_pipeline(
                params=params,
                pipeline_type='StableDiffusionXLImg2ImgPipeline',
                pipeline_model=model,
                vae_model=self.vae_model,
                overrides=overrides,
                output_type='pil'
            )
        return decoded_image

    def run_cascade_pipeline(
            self,
            params,
            overrides = {},
            show_image = True,
    ):
        
        self.ws.ud.progress = 0

        with torch.no_grad(): 

            # PRIOR PARAMS
            prior_params = {
                'prompt': params['prompt'],
                'negative_prompt': params['negative_prompt'],
                'num_inference_steps': params['inference_steps'],
                'guidance_scale': params['cfg_scale'],
                'height': params['height'],
                'width': params['width'],
                'num_images_per_prompt': 1,
            }
            # CALCULATED SETTINGS
            prior_params['num_inference_steps'] = int(params['inference_steps'] * params['steps_multiplier'])

            # DECODER PARAMS
            decoder_params = {
                'prompt': params['prompt'],
                'negative_prompt': params['negative_prompt'],
                'guidance_scale': 0.0,
                'output_type': "pil",
                'num_inference_steps': 10,
            }
            
            prior_type = 'StableCascadeCombinedPipeline'
            prior_model = 'stabilityai/stable-cascade-prior'
            decoder_type = 'StableCascadeDecoderPipeline'
            decoder_model = 'stabilityai/stable-cascade'
  
            self.ws.ud.progress_text = 'Loading pipeline...'
            prior_model_params = {
                'torch_dtype': torch.bfloat16,
                'add_watermarker': False,
            }
            decoder_model_params = {
                'torch_dtype': torch.float16,
                'add_watermarker': False,
            }

            # LOAD PRIOR
            try:
                unet = StableCascadeUnet.from_pretrained(prior_model, torch_dtype=torch.float16, subfolder="unet", in_channels=64, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
                self.pipe = globals()[prior_type].from_pretrained(decoder_model, unet=unet, **prior_model_params)
                self.pipe.to("cuda")

            except Exception as e:
                print(f"UD: Error occurred in loading the pipeline:\n\n{e}")
                self.unload()
                return None

            # RUN PRIOR
            try:
                prior_output = self.pipe(
                    **prior_params,
                ).images
            except Exception as e:
                print(f"UD: Error occurred while running the pipeline:\n\n{e}")
                self.unload()
                return None
            
            self.unload()
            
            # LOAD DECODER
            # try:
            #     self.pipe = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16)
            #     # self.pipe = globals()[decoder_type].from_pretrained(decoder_model, **decoder_model_params)
            #     self.pipe.to("cuda")

            # except Exception as e:
            #     print(f"UD: Error occurred in loading the pipeline:\n\n{e}")
            #     self.unload()
            #     return None
            
            # # RUN DECODER
            # try:
            #     decoder_output = self.pipe(
            #         image_embeddings=prior_output.image_embeddings.half(),
            #         **decoder_params,
            #     ).images

            # except Exception as e:
            #     print(f"UD: Error occurred while running the pipeline:\n\n{e}")
            #     self.unload()
            #     return None

            # RETURN IMAGE
            return prior_output[0]  

    def run_sdxl_pipeline(
            self,
            params,
            pipeline_type,
            pipeline_model, 
            vae_model = None,
            controlnet_models = [],
            t2i_models = [],
            overrides = {},
            show_image = True,
            output_type = 'latent',
        ):

        self.ws.ud.progress = 0

        with torch.no_grad(): 
            # Initializing dict with common parameters
            pipe_params = {
                'prompt': params['prompt'],
                'negative_prompt': params['negative_prompt'],
                'num_inference_steps': params['inference_steps'],
                'guidance_scale': params['cfg_scale'],
            }
            pipe_params.update(overrides)

            # CHANGES FOR SPECIFIC MODELS
            if pipeline_model in ['stablediffusionapi/NightVision_XL']:
                vae_model = None

            # INITIALIZE PIPE IF NEEDED
            if self.loaded_model != pipeline_model or self.loaded_model_type != pipeline_type or self.loaded_vae != vae_model or self.loaded_controlnets != controlnet_models or self.loaded_t2i != t2i_models:
                
                self.ws.ud.progress_text = 'Loading pipeline...'
                model_params = {
                    'torch_dtype': torch.float16,
                    'add_watermarker': False,
                }

                # LOAD CONTROLNET
                if controlnet_models:
                    model_params['controlnet'] = [self.create_controlnet(model) for model in controlnet_models]

                # LOAD T2I_ADAPTER
                if t2i_models:
                    if len(t2i_models) == 1:
                        model_params['adapter'] = self.create_t2i(t2i_models[0])
                    else:
                        model_params['adapter'] = MultiAdapter([self.create_t2i(model) for model in t2i_models])

                # LOAD VAE
                if vae_model:
                    model_params['vae'] = AutoencoderKL.from_pretrained( vae_model, torch_dtype=torch.float16 ).to("cuda")
                
                try:
                    try:
                        self.pipe = globals()[pipeline_type].from_pretrained(pipeline_model, **model_params, variant= 'fp16')
                        print("Loaded fp16 weights")
                    except Exception as e2:
                        print(f"fp16 variant not available. Using fp32.")
                        self.pipe = globals()[pipeline_type].from_pretrained(pipeline_model, **model_params)

                    self.pipe.to("cuda")
                    self.pipe.enable_vae_tiling()

                except Exception as e:
                    print(f"UD: Error occurred in loading the pipeline:\n\n{e}")
                    self.unload()
                    return None

                self.loaded_vae = vae_model
                self.loaded_model = pipeline_model
                self.loaded_model_type = pipeline_type
                self.loaded_controlnets = controlnet_models
                self.loaded_t2i = t2i_models
                del model_params

            # CALCULATED SETTINGS
            pipe_params['num_inference_steps'] = int(pipe_params['num_inference_steps'] * params['steps_multiplier'])

            # SPECIAL SETTINGS FOR SOME MODELS
            if 'sdxl-turbo' in pipeline_model:
                pipe_params['guidance_scale'] = 0

            print(self.loaded_model + ' ' + self.loaded_model_type)

            if pipeline_type not in ['StableDiffusionXLAdapterPipeline']:
                pipe_params['callback_on_step_end'] = self.pipe_callback

            # RUN STABLE DIFFUSION
            try:
                latent_image = self.pipe(
                    **pipe_params,
                    output_type='latent',
                ).images
            except Exception as e:
                print(f"UD: Error occurred while running the pipeline:\n\n{e}")
                self.unload()
                return None

            if show_image == True or output_type == 'pil':
                with torch.no_grad():
                    image = self.pipe.vae.decode(latent_image / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                    decoded_image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
            
                    decoded_image.save(params['temp_image_filepath'])

            if output_type == 'latent':
                return latent_image
            elif output_type == 'pil':
                return decoded_image
            
    def pipe_callback(self, pipe, step_index, timestep, callback_kwargs):

        if self.ws.ud.stop_process == 1:
            self.ws.ud.stop_process = 0
            raise Exception("Inference cancelled.") ## No cleaner way found

        self.ws.ud.progress = int((step_index + 1) / pipe.num_timesteps * 100)
        self.ws.ud.progress_text = f'Step {step_index + 1} / {pipe.num_timesteps}'

        for screen in self.ws.screens:
            for area in screen.areas:
                area.tag_redraw()

        return callback_kwargs
    
    def determine_pipeline_type(self, params, init_image, mask_image):
        if 'controlnet_model' in params:
            if mask_image and init_image:
                return 'StableDiffusionXLControlNetInpaintPipeline'
            return 'StableDiffusionXLControlNetImg2ImgPipeline' if init_image else 'StableDiffusionXLControlNetPipeline'
        elif 't2i_model' in params:
            return 'StableDiffusionXLAdapterPipeline'
        else:
            if mask_image and init_image:
                return 'StableDiffusionXLInpaintPipeline'
            return 'StableDiffusionXLImg2ImgPipeline' if init_image else 'StableDiffusionXLPipeline'
        
    def create_controlnet(self, controlnet_model):
        model = None

        if CONTROLNET_MODELS[controlnet_model]['model_type'] == 'diffusers':
            try:
                model = ControlNetModel.from_pretrained(controlnet_model, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to("cuda")     
                return model
            except Exception as e:
                pass
            
            try:
                model = ControlNetModel.from_pretrained(controlnet_model, use_safetensors=True, torch_dtype=torch.float16).to("cuda")
                return model
            except Exception as e:
                pass

            try:
                model = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16).to("cuda")
                return model
            except Exception as e:
                pass

        if not model:
            print("Failed to load Controlnet!")
            
        return model
   
    def create_t2i(self, t2i_model):
        model = None

        try:
            model = T2IAdapter.from_pretrained(t2i_model, torch_dtype=torch.float16, variant="fp16").to("cuda")
            return model
        except Exception as e:
            pass

        if not model:
            print("Failed to load T2I Adapter!")

        return model

    def unload(self):

        for item in ['pipe']:
            if hasattr(self, item):
                # getattr(self, item).to('cpu')
                delattr(self, item)

        torch.cuda.empty_cache()

        self.loaded_model = None
        self.loaded_model_type = None
        self.loaded_vae = None
        self.loaded_controlnets = None
        self.loaded_t2i = None

        print("GPU cache has been cleared.")