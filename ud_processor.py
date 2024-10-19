
import os, platform

from diffusers import FluxPipeline, T2IAdapter, MultiAdapter, EDMDPMSolverMultistepScheduler, DPMSolverMultistepScheduler, StableDiffusionXLControlNetPipeline, DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLAdapterPipeline, StableDiffusionUpscalePipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL
import numpy as np
import torch
from PIL import Image, ImageEnhance
from realesrgan_ncnn_py import Realesrgan

from . import gpudetector
from .constants import CONTROLNET_MODELS
from .pipelines import pipeline_settings

current_dir = os.path.dirname(os.path.realpath(__file__))

# Install opencv-python-headless instead of regular opencv-python! Or you'll run into xcb conflicts

def round_to_nearest(n):
    if n - int(n) < 0.5:
        return int(n)
    else:
        return int(n) + 1
    
def blender_image_to_pil(blender_image):
    if blender_image is None:
        raise ValueError("No Blender image provided")

    pixels = np.array(blender_image.pixels[:]) 
    size = blender_image.size[0], blender_image.size[1]

    pixels = np.reshape(pixels, (size[1], size[0], 4))
    pixels = np.flip(pixels, axis=0)
    pixels = (pixels * 255).astype(np.uint8)

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

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.has_rocm:
        return torch.device('hip')
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class UD_Processor():
    prompt_adds = ", highly detailed, beautiful, 4K, photorealistic, high resolution"
    negative_prompt_adds = ", text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, bad, distortion, twisted, grainy, duplicate, error, pixelated, fake, glitch, overexposed, bad-contrast"
    vae_model = "madebyollin/sdxl-vae-fp16-fix"

    upscale_strength = 0.35
    upscaling_rate = 2
    upscaling_steps = 10

    loaded_model = None
    loaded_model_type = None
    loaded_vae = None
    loaded_controlnets = None
    loaded_t2i = None

    device = get_device()

    manager = None

    def run(self, params, manager):
        self.manager = manager

        target_width, target_height = (round((params[dim] * params['scale'] / 100) / 16) * 16 for dim in ['width', 'height'])

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

        if init_image:
            init_image = init_image.convert('RGB')

        controlnet_image = [blender_image_to_pil(slot).resize((target_width, target_height)).convert("RGB") for slot in params['controlnet_image_slot']] if 'controlnet_image_slot' in params else None
        t2i_image = [blender_image_to_pil(slot).resize((target_width, target_height)).convert("RGB") for slot in params['t2i_image_slot']] if 't2i_image_slot' in params else None
            
        pipeline_type = self.determine_pipeline_type(params, init_image, mask_image)

        # Define a dictionary of potential parameter assignments with lambdas for conditional logic
        param_mapping = {
            'prompt': lambda: params['prompt'] + self.prompt_adds,
            'width': lambda: target_width,
            'height': lambda: target_height,
            'generator': lambda: torch.manual_seed(params["seed"]),
            'num_inference_steps': lambda: round_to_nearest(params['inference_steps'] / (params['denoise_strength'] if init_image else 1)),
            'guidance_scale': lambda: params['cfg_scale'],
            'negative_prompt': lambda: params['negative_prompt'] + self.negative_prompt_adds,
            'image': lambda: next(
                (img for img in (t2i_image, init_image, controlnet_image) if img is not None),
                None
            ),
            'mask_image': lambda: mask_image,
            'strength': lambda: params['denoise_strength'],
            'control_image': lambda: params.get('controlnet_image'),
            'controlnet_conditioning_scale': lambda: params['controlnet_factor'],
            'adapter_conditioning_scale': lambda: params['t2i_factor'] if len(params['t2i_model']) > 1 else params['t2i_factor'][0],
        }

        pipe_params = {}
        for key in pipeline_settings[pipeline_type]:
            if key in param_mapping:
                value = param_mapping[key]()
                if value is not None:  
                    pipe_params[key] = value

        image = self.run_pipeline(
            params=params,
            pipeline_type=pipeline_type,
            pipeline_model=params['model'],
            vae_model=self.vae_model,
            controlnet_models=params.get('controlnet_model', []),
            t2i_models=params.get('t2i_model', []),
            pipe_params=pipe_params,
        )

        if image is not None:
            return image
        
    def upscale(self, params, manager): 
        self.manager = manager

        image = Image.open(params['temp_image_filepath'])

        current_width = round_to_nearest(params['width']/16)*16
        current_height = round_to_nearest(params['height']/16)*16

        if params['mode'] == 'upscale_re':
            
            self.manager.set_progress(0)
            self.manager.set_progress_text('Resizing with Realesrgan ...')

            # Resize to 4x using realesrgan
            realesrgan = Realesrgan(gpuid = gpudetector.get_dedicated_gpu(), model = 4)
            image = realesrgan.process_pil(image)
            realesrgan = None
            upscaled_image = image.resize((current_width * 2, current_height * 2), Image.Resampling.LANCZOS)
            contrast=1.1

        elif params['mode'] == 'upscale_sd':
            # Resize to 4x using stable-diffusion-x4-upscaler
            self.unload()   
            model_id = "stabilityai/stable-diffusion-x4-upscaler"
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipe = self.pipe.to(self.device)
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
            decoded_image = self.run_pipeline(
                params=params,
                pipeline_type='StableDiffusionXLImg2ImgPipeline',
                pipeline_model=model,
                vae_model=self.vae_model,
                pipe_params=overrides,
            )
        return decoded_image

    def run_pipeline(
            self,
            params,
            pipeline_type,
            pipeline_model, 
            vae_model = None,
            controlnet_models = [],
            t2i_models = [],
            pipe_params = {},
        ):

        self.manager.set_progress(0)

        with torch.no_grad(): 
            # CHANGES FOR SPECIFIC MODELS
            if pipeline_model not in ['stabilityai/stable-diffusion-xl-base-1.0']:
                vae_model = None

            # INITIALIZE PIPE IF NEEDED
            if self.loaded_model != pipeline_model or self.loaded_model_type != pipeline_type or self.loaded_vae != vae_model or self.loaded_controlnets != controlnet_models or self.loaded_t2i != t2i_models:
                
                self.unload()
                self.manager.set_progress_text('Loading pipeline...')

                model_params = {
                    'torch_dtype': torch.float16,
                }

                if params['pipeline_type'] == 'SDXL':
                    model_params['add_watermarker'] = False

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
                    model_params['vae'] = AutoencoderKL.from_pretrained( vae_model, torch_dtype=torch.float16 ).to(self.device)
                
                try:
                    try:
                        self.pipe = globals()[pipeline_type].from_pretrained(pipeline_model, **model_params, variant= 'fp16')
                        print("Loaded fp16 weights")
                    except Exception as e2:
                        print(f"fp16 variant not available. Using fp32.")
                        self.pipe = globals()[pipeline_type].from_pretrained(pipeline_model, **model_params)

                    if params['pipeline_type'] == 'SDXL':
                        self.pipe.to(self.device)
                        self.pipe.enable_vae_tiling()
                    elif params['pipeline_type'] == 'FLUX':
                        self.pipe.enable_sequential_cpu_offload()
                        self.pipe.vae.enable_slicing()
                        self.pipe.vae.enable_tiling()

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

            print(self.loaded_model + ' ' + self.loaded_model_type)

            if pipeline_type not in ['StableDiffusionXLAdapterPipeline']:
                pipe_params['callback_on_step_end'] = self.pipe_callback

            if pipeline_model in ['playgroundai/playground-v2.5-1024px-aesthetic']:
                self.pipe.scheduler = EDMDPMSolverMultistepScheduler()

            # RUN DIFFUSION
            try:
                decoded_image = self.pipe(
                    **pipe_params,
                    output_type='pil',
                ).images[0]
            except Exception as e:
                print(f"UD: Error occurred while running the pipeline:\n\n{e}")
                self.unload()
                return None
            
            decoded_image.save(params['temp_image_filepath'])

            return decoded_image
            
    def pipe_callback(self, pipe, step_index, timestep, callback_kwargs):
        if self.manager.stop_process() == 1:
            self.manager.set_stop_process(0)
            raise Exception("Inference cancelled.") ## No cleaner way found

        self.manager.set_progress(int((step_index + 1) / pipe.num_timesteps * 100))
        self.manager.set_progress_text(f'Step {step_index + 1} / {pipe.num_timesteps}')

        self.manager.redraw()

        return callback_kwargs
    
    def determine_pipeline_type(self, params, init_image, mask_image):
        if params['pipeline_type'] == 'SDXL':
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
        elif params['pipeline_type'] == 'FLUX':
            return 'FluxPipeline'
        
    def create_controlnet(self, controlnet_model):
        if CONTROLNET_MODELS[controlnet_model]['model_type'] == 'diffusers':
            for kwargs in [{"variant": "fp16", "use_safetensors": True}, {"use_safetensors": True}, {}]:
                try:
                    return ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16, **kwargs).to(self.device)
                except Exception:
                    continue

        print("Failed to load Controlnet!")
        return None
   
    def create_t2i(self, t2i_model):
        model = None

        try:
            model = T2IAdapter.from_pretrained(t2i_model, torch_dtype=torch.float16, variant="fp16").to(self.device)
            return model
        except Exception as e:
            pass

        if not model:
            print("Failed to load T2I Adapter!")

        return model

    def unload(self):

        self.manager.set_progress_text('Unloading loaded model...')

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

        self.manager.set_progress_text('Unloaded')
        print("GPU cache has been cleared.")