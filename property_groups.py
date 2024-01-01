import bpy

SD_MODELS = [
    ('stablediffusionapi/NightVision_XL','NightVision_XL', ""),
    ('stabilityai/stable-diffusion-xl-base-1.0','SDXL Base', ""),
    ('segmind/SSD-1B','SSD-1B', ""),
    ('stabilityai/sdxl-turbo','SDXL Turbo', ""),
    ('Lykon/dreamshaper-xl-turbo','Dreamshaper XL Turbo',''),
    ('Lykon/dreamshaper-xl-1-0','Dreamshaper XL 1.0',''),
    ('stablediffusionapi/juggernaut-xl-v7','Juggernaut XL v7',''),
    ('playgroundai/playground-v2-1024px-aesthetic','Playground V2 Aesthetic', ""),
]

CONTROLNET_MODELS = [
    ('diffusers/controlnet-depth-sdxl-1.0', 'controlnet-depth-sdxl-1.0', ''),
    ('diffusers/controlnet-depth-sdxl-1.0-small', 'controlnet-depth-sdxl-1.0-small', ''),
    ('diffusers/controlnet-depth-sdxl-1.0-mid', 'controlnet-depth-sdxl-1.0-mid', ''),
    ('diffusers/controlnet-canny-sdxl-1.0', 'controlnet-canny-sdxl-1.0', ''),
    ('diffusers/controlnet-canny-sdxl-1.0-small', 'controlnet-canny-sdxl-1.0-small', ''),
    ('diffusers/controlnet-canny-sdxl-1.0-mid', 'controlnet-canny-sdxl-1.0-mid', ''),
    ('Nacholmo/controlnet-qr-pattern-sdxl', 'controlnet-qr-pattern-sdxl', ''),
    ('SargeZT/sdxl-controlnet-seg', 'sdxl-controlnet-seg', ''),
    ('SargeZT/controlnet-sd-xl-1.0-softedge-dexined', 'controlnet-sd-xl-1.0-softedge-dexined', '')
]

class ControlNetListItem(bpy.types.PropertyGroup):
    controlnet_model: bpy.props.EnumProperty(
        name='',
        items=CONTROLNET_MODELS
    )
    controlnet_image_slot: bpy.props.PointerProperty(
        name='',
        type=bpy.types.Image
    )
    controlnet_factor: bpy.props.FloatProperty(
        name='',
        min=0.0, max=1.0, step=0.05, default=0.5
    )
    
class MY_UL_ControlNetList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row()

        delete_op = row.operator("controlnet.remove_item", text="", icon='X')
        delete_op.item_index = index

        row.prop(item, "controlnet_model")
        row.prop(item, "controlnet_image_slot")
        
        col = row.column()
        col.scale_x = 0.6
        col.prop(item, "controlnet_factor")


class UDPropertyGroup(bpy.types.PropertyGroup):
    model: bpy.props.EnumProperty(items=SD_MODELS, name="Model")
    prompt: bpy.props.StringProperty(name="Prompt", default="A close up of a cat with sunglasses driving a ferrari, golden hour")
    negative_prompt: bpy.props.StringProperty(name="Negative Prompt")
    scale: bpy.props.IntProperty(
        name='Scale',
        soft_max=1000,
        default=50,
        min=0,
    )
    width: bpy.props.IntProperty(
        name='Width',
        soft_max=10000,
        default=1920,
        min=0,
    )
    height: bpy.props.IntProperty(
        name='Height',
        soft_max=10000,
        default=1080,
        min=0,
    )
    seed: bpy.props.IntProperty(
        name='Seed',
        soft_max=99999,
        default=0,
        soft_min=0,
    )
    inference_steps: bpy.props.IntProperty(
        name='Inference steps',
        soft_max=100,
        default=50,
        min=1,
    )
    cfg_scale: bpy.props.FloatProperty(
        name='CFG scale',
        soft_max=100,
        default=5,
        min=0,
        precision=1,
    )
    refiner_strength: bpy.props.FloatProperty(
        name='Refiner strength',
        max=1,
        default=0.3,
        min=0,
        precision=2,
    )
    high_noise_frac: bpy.props.FloatProperty(
        name='High noise fraction',
        max=1,
        default=1,
        min=0,
        precision=2,
    )
    init_image_slot: bpy.props.PointerProperty(
        name="Init image slot",
        description="Enter the slot for an init image to condition the generation",
        type=bpy.types.Image
    )
    init_mask_slot: bpy.props.PointerProperty(
        name="Mask image slot",
        description="Enter the slot for a masking image for the inpainting generation",
        type=bpy.types.Image
    )
    denoise_strength: bpy.props.FloatProperty(
        name='Denoise Strength',
        max=1,
        default=0.4,
        min=0,
        precision=2,
    )

    controlnet_list : bpy.props.CollectionProperty(type=ControlNetListItem)
    controlnet_list_index : bpy.props.IntProperty(
        name="Search Result",
        default=-1,
        update=lambda self, context: setattr(self, 'controlnet_list_index', -1)
    )

    running : bpy.props.BoolProperty(
        name="is running",
        default=0
    )