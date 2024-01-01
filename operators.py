import bpy, os, tempfile, threading, random
import numpy as np

from . import property_groups as pg
from . import ud_processor as ud

# Create a temporary file path that works on both Windows and Unix-like systems
temp_image_filepath = os.path.join(tempfile.gettempdir(), "temp.png")
print("Temporary file path:", temp_image_filepath)

worker = ud.UD_Processor()

class Run_UD(bpy.types.Operator):
    bl_idname = "image.run_ud"
    bl_label = "Run Unexpected Diffusion"

    mode: bpy.props.StringProperty()

    def ud_task(self, parameters, image_area, ws):

        try:
            result = worker.run(params = parameters)

            if result:
                image = bpy.data.images.load(parameters['temp_image_filepath'])
                image.name = parameters['prompt'][:57] + "-" + str(parameters['seed'])
                image_area.spaces.active.image = image

        except Exception as e:
            self.report({'INFO'}, f"Error occurred: {e}")

        ws.ud.running = 0

    def ud_upscale_task(self, parameters, image_area, ws):
        try:
            space = image_area.spaces.active

            if space.image:
                parameters['width'] = space.image.size[0]
                parameters['height'] = space.image.size[1]
            
                original_view_transform = bpy.context.scene.view_settings.view_transform
                bpy.context.scene.view_settings.view_transform = 'Raw'
                bpy.data.images[space.image.name].save_render(parameters['temp_image_filepath'])
                bpy.context.scene.view_settings.view_transform = original_view_transform

                worker.upscale(params = parameters)

                image = bpy.data.images.load(parameters['temp_image_filepath'])
                image.name = parameters['prompt'][:57] + "-" + str(parameters['seed'])
                image_area.spaces.active.image = image
                
        except Exception as e:
            self.report({'INFO'}, f"Error occurred: {e}")

        finally:
            ws.ud.running = 0

    def execute(self, context):
        areas = bpy.context.screen.areas
        ws = bpy.context.workspace

        ws.ud.running = 1

        # Prepare parameters
        parameters = {prop.identifier: getattr(ws.ud, prop.identifier) 
                   for prop in pg.UDPropertyGroup.bl_rna.properties 
                   if not prop.is_readonly}
        
        parameters['temp_image_filepath'] = temp_image_filepath

        if ws.ud.seed == 0:
            parameters['seed'] = random.randint(1, 99999)

        for item in ws.ud.controlnet_list:
            if item.controlnet_image_slot and item.controlnet_factor > 0:
                for entry in ['controlnet_model','controlnet_image_slot','controlnet_factor']:
                    if not parameters.get(entry):
                        parameters[entry]=[]
                    parameters[entry].append(getattr(item, entry))

        parameters['mode'] = self.mode
        print(parameters)
        
        for area in areas:
            if area.type == 'IMAGE_EDITOR':
                image_area = area

        if self.mode == 'generate': 
            thread = threading.Thread(target=self.ud_task, args=[parameters, image_area, ws])
        elif self.mode in ['upscale_sd','upscale_re']:
            thread = threading.Thread(target=self.ud_upscale_task, args=[parameters, image_area, ws])
        
        thread.start()

        return {'FINISHED'}
    
class Unload_UD(bpy.types.Operator):
    bl_idname = "image.unload_ud"
    bl_label = "Release memory"

    def execute(self, context):
        worker.unload()
        return {'FINISHED'}
    
class Controlnet_AddItem(bpy.types.Operator):
    bl_idname = "controlnet.add_item"
    bl_label = "Add ControlNet Item"

    def execute(self, context):
        ws = context.workspace  # Replace with your actual data path
        ws.ud.controlnet_list.add()  # Adjust this line based on how you access your list

        return {'FINISHED'}
    
class Controlnet_RemoveItem(bpy.types.Operator):
    bl_idname = "controlnet.remove_item"
    bl_label = "Remove Controlnet"

    item_index: bpy.props.IntProperty()  

    def execute(self, context):
        ws = context.workspace
        ws.ud.controlnet_list.remove(self.item_index)
        
        return {'FINISHED'}