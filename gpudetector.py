import collections
import collections.abc

# Compatibility layer for collections.Iterable in Python 3.10
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

import vulkan as vk

def get_nvidia_gpu():
    # Initialize Vulkan
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName='Vulkan GPU List',
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName='No Engine',
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0)

    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info)

    instance = vk.vkCreateInstance(create_info, None)

    # Enumerate Physical Devices (GPUs)
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)

    for i, device in enumerate(physical_devices):
        properties = vk.vkGetPhysicalDeviceProperties(device)
        device_name = properties.deviceName
        if "NVIDIA" in device_name:
            gpu_id = i
            break

    # Clean up
    vk.vkDestroyInstance(instance, None)
    return gpu_id