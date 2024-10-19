import bpy
from . import PG_NAME_LC, blender_globals, dependencies_installed

class MY_UL_ControlList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)

        row = layout.row()

        mode = pg.control_mode

        delete_op = row.operator(f"control.remove_item", text="", icon='X')
        delete_op.item_index = index

        row.prop(item, f"{mode}_model")
        row.prop(item, f"{mode}_image_slot")
        
        col = row.column()
        col.scale_x = 0.6
        col.prop(item, f"{mode}_factor")

class UDPanel(bpy.types.Panel):
    """Creates a Panel in the Image Editor"""
    bl_idname = f"SCENE_PT_{PG_NAME_LC}"
    bl_label = "Unexpected Diffusion"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Unexpected Diffusion'

    def draw(self, context):
        layout = self.layout
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)
        
        row = layout.row()
        row.label(text="Unexpected Diffusion", icon='UV')

        if not dependencies_installed:
            row = layout.row()
            row.label(text="Dependencies not installed!")
            return

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
                    item == 'refiner_strength' and pg.high_noise_frac == 1  # Check for 'refiner_strength' with high noise
                    or item in ['denoise_strength', 'init_mask_slot'] and not pg.init_image_slot  # Check for 'denoise_strength' or 'init_mask_slot' without an init image
                    or item in ['init_image_slot', 'denoise_strength', 'init_mask_slot'] and pg.control_mode == 't2i'
                ):
                    continue

                row.prop(pg, item)
                has_item = True
            
            if has_item: 
                row = layout.row()

        if pg.control_mode == 't2i':
            current_list = 't2i_list'
            current_mode_xlat = 'T2I Adapter'
            switch_to = 'controlnet'
            switch_to_xlat = 'Controlnet'
        elif pg.control_mode == 'controlnet':
            current_list = 'controlnet_list'
            
            current_mode_xlat = 'Controlnet'
            switch_to = 't2i'
            switch_to_xlat = 'T2I Adapter'

        if len(getattr(pg, current_list)) > 0:
            row.template_list("MY_UL_ControlList", f"{current_mode_xlat}s",
                pg, current_list,
                pg, "control_list_index"
                )

        row = layout.row()
        row.operator(f"{PG_NAME_LC}.control_add_item", icon='ADD', text=f'Add {current_mode_xlat}')
        row.operator(f"{PG_NAME_LC}.control_mode", text=f'Switch to {switch_to_xlat}', icon='ARROW_LEFTRIGHT').switch_mode=switch_to

        row = layout.row()
        row = row.separator(factor = 2)
        if pg.running == 0:
            row = layout.row()
            row.operator(f"{PG_NAME_LC}.run_ud", text="Run Unexpected Diffusion", icon='IMAGE').mode='generate'
            row.operator(f"{PG_NAME_LC}.unload_ud", text="Release Memory", icon='UNLINKED')
            row = layout.row()
            row.operator(f"{PG_NAME_LC}.run_ud", text="Run Light 2x Upscaler", icon='ZOOM_IN').mode='upscale_re'
            row.operator(f"{PG_NAME_LC}.run_ud", text="Run Heavy 2x Upscaler", icon='ZOOM_IN').mode='upscale_sd'

        if pg.running == 1:
            row = layout.row()
            row.prop(pg, "progress", text=pg.progress_text, slider=True)
            row.enabled = False
            row = layout.row()
            row.operator(f"{PG_NAME_LC}.stop_ud", text="Stop Generation", icon='QUIT')

        row = layout.row()
        row = row.separator(factor = 2)
        row = layout.row()
        row.label(text="Utilities", icon='TOOL_SETTINGS')

        row = layout.row()
        row.operator(f"{PG_NAME_LC}.generate_projected_uvs", icon='MOD_UVPROJECT')

        row = layout.row()
        depth_operator_3d = row.operator(f"{PG_NAME_LC}.generate_map", text="Depth from 3D", icon='IMAGE_ZDEPTH')
        depth_operator_3d.mode='depth'
        depth_operator_3d.target='3d'

        row = layout.row()
        canny_operator_3d = row.operator(f"{PG_NAME_LC}.generate_map", text="Canny from 3D", icon='IMAGE_ZDEPTH')
        canny_operator_3d.mode='canny'
        canny_operator_3d.target='3d'

        canny_operator_image = row.operator(f"{PG_NAME_LC}.generate_map", text="Canny from Image", icon='IMAGE_ZDEPTH')
        canny_operator_image.mode='canny'
        canny_operator_image.target='image'
        row.prop(pg, "canny_strength", text="Strength")
