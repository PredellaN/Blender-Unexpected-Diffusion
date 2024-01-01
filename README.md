# Unexpected Diffusion Addon for Blender
![cat](https://github.com/PredellaN/Blender-Unexpected-Diffusion/assets/75480205/ba754c19-526b-4775-b60c-01cb6d4260ca)

## Overview
Unexpected Diffusion is an addon for Blender that integrates SDXL into the Image Editor panel.

## Installation
1. Download the addon from the GitHub repository.
2. Unzip
3. Install the dependencies using either the linux or windows script (windows script is untested).
4. Zip back
5. Open Blender and go to `Edit > Preferences > Add-ons`.
6. Click `Install` and navigate to the zip file.
7. Check the box next to the addon name to enable it.

## Features
Basic SDXL functionality including img2img and inpainting, two custom upscaling techniques. Limited number of models available (i try to keep the list short and tested)

## Usage
After installation, open the Image Editor in Blender. You'll find the "Misc" panel on the right side (use `N` if it's not visible). Adjust the settings according to your needs, and click "Run Unexpected Diffusion" to start generating images.
Tested only with NVIDIA GPU (commits welcome for AMD / Intel / Apple GPUs) and only on Linux (it should work out of the box on windows as well)

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
- Controlnet
- Easier inpainting
- Depth to controlnet
