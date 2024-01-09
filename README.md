# Unexpected Diffusion - Stable Diffusion XL Addon for Blender
![taormina at night(94846)](https://github.com/PredellaN/Blender-Unexpected-Diffusion/assets/75480205/5e0e83c5-1d87-47f7-851c-a714d8913f25)

## Overview
Unexpected Diffusion is an addon for Blender that integrates SDXL into the Image Editor panel. The workflow and memory management is optimized to be able to generate very large images with the help of init images, controlnets, and my custom upscaling workflows.

![tiletest_3x](https://github.com/PredellaN/Blender-Unexpected-Diffusion/assets/75480205/b5af48cf-1792-4074-a01d-d83e68f7970b)

## Installation
1. Download the addon from the GitHub repository.
2. Unzip
3. Install the dependencies using either the .sh linux or .bat windows script (windows script is untested, please report if it works as intended).
4. Zip back
5. Open Blender and go to `Edit > Preferences > Add-ons`.
6. Click `Install` and navigate to the zip file.
7. Check the box next to the addon name to enable it.

## Features
Basic SDXL functionality including img2img and inpainting, two custom upscaling techniques. Limited number of models available (i try to keep the list short and tested)

## Usage
After installation, open the Image Editor in Blender. You'll find the "Misc" panel on the right side (use `N` if it's not visible). Adjust the settings according to your needs, and click "Run Unexpected Diffusion" to start generating images.
Tested only with NVIDIA GPU (commits welcome for AMD / Intel / Apple GPUs) and only on Linux (it should work out of the box on windows as well)

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
- easier inpainting
- utility to generate also segmentation map
- seamless generation
- batch generation
- progress bar
- low memory warnings (using gpudetector info and heuristics)
