import bpy

class MY_UL_ControlList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row()

        mode = context.workspace.ud.control_mode

        delete_op = row.operator(f"control.remove_item", text="", icon='X')
        delete_op.item_index = index

        row.prop(item, f"{mode}_model")
        row.prop(item, f"{mode}_image_slot")
        
        col = row.column()
        col.scale_x = 0.6
        col.prop(item, f"{mode}_factor")

class UDPanel(bpy.types.Panel):
    """Creates a Panel in the Image Editor"""
    bl_idname = "UD_PT_Gen"
    bl_label = "Unexpected Diffusion"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'

    def draw(self, context):
        layout = self.layout
        ws = context.workspace
        # ws.ud.running = 0
        
        row = layout.row()
        row.label(text="Unexpected Diffusion", icon='UV')

        row = layout.row()
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
            
            has_item = False
            for item in item_list:
                if (
                    item == 'refiner_strength' and ws.ud.high_noise_frac == 1  # Check for 'refiner_strength' with high noise
                    or item in ['denoise_strength', 'init_mask_slot'] and not ws.ud.init_image_slot  # Check for 'denoise_strength' or 'init_mask_slot' without an init image
                    or item in ['init_image_slot', 'denoise_strength', 'init_mask_slot'] and ws.ud.control_mode == 't2i'
                ):
                    continue

                row.prop(ws.ud, item)
                has_item = True
            
            if has_item: 
                row = layout.row()

        if ws.ud.control_mode == 't2i':
            current_list = 't2i_list'
            current_mode_xlat = 'T2I Adapter'
            switch_to = 'controlnet'
            switch_to_xlat = 'Controlnet'
        elif ws.ud.control_mode == 'controlnet':
            current_list = 'controlnet_list'
            current_mode_xlat = 'Controlnet'
            switch_to = 't2i'
            switch_to_xlat = 'T2I Adapter'

        if len(getattr(ws.ud, current_list)) > 0:
            row.template_list("MY_UL_ControlList", f"{current_mode_xlat}s",
                ws.ud, current_list,
                ws.ud, "control_list_index"
                )

        row = layout.row()
        row.operator("control.add_item", icon='ADD', text=f'Add {current_mode_xlat}')
        row.operator("control.control_mode", text=f'Switch to {switch_to_xlat}', icon='ARROW_LEFTRIGHT').switch_mode=switch_to

        row = layout.row()
        row = row.separator(factor = 2)
        if ws.ud.running == 0:
            row = layout.row()
            row.operator("image.run_ud", text="Run Unexpected Diffusion", icon='IMAGE').mode='generate'
            row.operator("image.unload_ud", text="Release Memory", icon='UNLINKED')
            row = layout.row()
            row.operator("image.run_ud", text="Run Light 2x Upscaler", icon='ZOOM_IN').mode='upscale_re'
            row.operator("image.run_ud", text="Run Heavy 2x Upscaler", icon='ZOOM_IN').mode='upscale_sd'

        if ws.ud.running == 1:
            row = layout.row()
            row.prop(ws.ud, "progress", text=ws.ud.progress_text, slider=True)
            row.enabled = False
            row = layout.row()
            row.operator("image.stop_ud", text="Stop Generation", icon='QUIT')

        row = layout.row()
        row = row.separator(factor = 2)
        row = layout.row()
        row.label(text="Utilities", icon='TOOL_SETTINGS')

        row = layout.row()
        row.operator("generate.projected_uvs", icon='MOD_UVPROJECT')

        row = layout.row()
        depth_operator_3d = row.operator("generate.map", text="Depth from 3D", icon='IMAGE_ZDEPTH')
        depth_operator_3d.mode='depth'
        depth_operator_3d.target='3d'

        row = layout.row()
        canny_operator_3d = row.operator("generate.map", text="Canny Edge from 3D", icon='IMAGE_ZDEPTH')
        canny_operator_3d.mode='canny'
        canny_operator_3d.target='3d'

        canny_operator_image = row.operator("generate.map", text="Canny Edge from Image", icon='IMAGE_ZDEPTH')
        canny_operator_image.mode='canny'
        canny_operator_image.target='image'