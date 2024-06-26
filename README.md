# Unexpected Diffusion - Stable Diffusion XL Addon for Blender

![taormina at night(94846)](https://github.com/PredellaN/Blender-Unexpected-Diffusion/assets/75480205/5e0e83c5-1d87-47f7-851c-a714d8913f25)

#### <p align="center"> Unexpected Diffusion is an addon for Blender that integrates SDXL into the Image Editor panel.</p>

![tiletest_3x](https://github.com/PredellaN/Blender-Unexpected-Diffusion/assets/75480205/b5af48cf-1792-4074-a01d-d83e68f7970b)

#### <p align="center">The workflow and memory management is optimized to be able to generate very large images with the help of init images, controlnets, t2i adapters, and my custom upscaling workflows.</p>

![wide_1_contrasted](https://github.com/PredellaN/Blender-Unexpected-Diffusion/assets/75480205/3cc3e48e-29d2-4f97-b9e8-7e60229660c9)

## Installation
1. Download the addon from the GitHub repository.
5. Open Blender and go to `Edit > Preferences > Add-ons`.
6. Click `Install` and navigate to the zip file.
7. Check the box next to the addon name to enable it.
8. Click on the "Install Dependencies" button and wait until finished 

## Features
Basic SDXL functionality including img2img and inpainting, two custom upscaling techniques. Limited number of models available (i try to keep the list short and tested)

## Requirements
An NVIDIA GPU with at least 8GB of memory

## Usage
After installation, open the Image Editor in Blender. You'll find the "Misc" panel on the right side (use `N` if it's not visible). Adjust the settings according to your needs, and click "Run Unexpected Diffusion" to start generating images.
Tested only with NVIDIA GPU on Linux (commits welcome for AMD / Intel / Apple GPUs).

To generate depth and canny maps with the utility, the view from the 3d viewport in the current tab will be used (generating the map will fail if there are no 3d viewports in the current tab) 

### Parameters:
- **Model**: Select the SDXL model you want to use for generation.
- **Prompt/Negative Prompt**: Enter a description of the image you want to create or elements you want to avoid.
- **Scale/Width/Height**: Define the dimensions and scale of the output image.
- **Seed**: Specify a seed for reproducible results.
- **Inference Steps**: Set the number of steps for the AI to refine the image.
- **And More**: Explore additional parameters for advanced customization.

## Contributing
Contributions are welcome, if you'd like to help improve Unexpected Diffusion, please fork the repository and submit a pull request with your changes.

## License
This project is licensed under GPLv3

## TODO
- TripoSR from image
- stable diffusion 3 as soon as available and sufficiently documented
- StableDiffusionXLInstantIDPipeline ( https://huggingface.co/InstantX/InstantID )
- easier inpainting
- seamless generation
- batch generation
- low memory warnings (using gpudetector info and heuristics)
- tileable texture generator from image
