import bpy

class UDPanel(bpy.types.Panel):
    """Creates a Panel in the Image Editor"""
    bl_idname = "UD_PT_Gen"
    bl_label = "Unexpected Diffusion"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'

    def draw(self, context):
        layout = self.layout
        ws = bpy.context.workspace
        # ws.ud.running = 0
        
        row = layout.row()
        row.label(text="Unexpected Diffusion", icon='UV')

        for item_list in [
            ['model'],
            ['prompt'],
            ['negative_prompt'],
            ['scale','width','height'],
            ['seed'],
            ['inference_steps','cfg_scale'],
            ['high_noise_frac','refiner_strength'],
            ['init_image_slot'],
            ['denoise_strength'],
            ['init_mask_slot']]:
            
            row = layout.row()
            for item in item_list:
                if (
                    item == 'refiner_strength' and ws.ud.high_noise_frac == 1  # Check for 'refiner_strength' with high noise
                    or item in ['denoise_strength', 'init_mask_slot'] and not ws.ud.init_image_slot  # Check for 'denoise_strength' or 'init_mask_slot' without an init image
                ):
                    continue
                row.prop(ws.ud, item)

        row = layout.row()
        if len(ws.ud.controlnet_list) > 0:
            row.template_list("MY_UL_ControlNetList", "Controlnets",
                ws.ud, "controlnet_list",
                ws.ud, "controlnet_list_index"
                )
            icon='ADD'
        else:
            icon='ORIENTATION_LOCAL'
        col = layout.column(align=True)
        col.operator("controlnet.add_item", icon=icon, text="Add ControlNet")

        row = layout.row()
        
        if ws.ud.running == 0:
            row.operator("image.run_ud", text="Run Unexpected Diffusion", icon='IMAGE').mode='generate'
            row.operator("image.unload_ud", text="Release Memory", icon='CANCEL')

        if ws.ud.running == 1:
            row.operator("image.run_ud", text="Unexpected Diffusion is running...", icon='IMAGE')
            row.enabled = False

        row = layout.row()
        if ws.ud.running == 0:
            row.operator("image.run_ud", text="Run Light 2x Upscaler", icon='ZOOM_IN').mode='upscale_re'
            row.operator("image.run_ud", text="Run Heavy 2x Upscaler", icon='ZOOM_IN').mode='upscale_sd'