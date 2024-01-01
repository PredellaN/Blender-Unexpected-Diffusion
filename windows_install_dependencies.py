import os, subprocess

def install_dependency(package_name, target_dir):
    """ Install a single package using pip. """
    print(f"Installing {package_name}...")
    subprocess.run(["pip", "install", "-t", target_dir, package_name], check=True)
    print(f"Installed {package_name} successfully.\n")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Folder for dependencies inside the script's directory
dependencies_dir = os.path.join(script_dir, "dependencies")

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(dependencies_dir):
    os.makedirs(dependencies_dir)
    print(f"Created directory {dependencies_dir}")

# List of dependencies to install
dependencies = ["numpy", "torch", "Pillow", "realesrgan-ncnn-py", "vulkan", 'omegaconf']

# Install each dependency
print("Starting installation of dependencies...")
for dep in dependencies:
    install_dependency(dep, dependencies_dir)

print("All dependencies installed successfully.")
