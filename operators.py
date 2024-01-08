import bpy, os, tempfile, threading, random, math
from bpy.types import Operator
import numpy as np
from PIL import Image

from . import property_groups as pg
from . import ud_processor as ud

# Create a temporary file path that works on both Windows and Unix-like systems
temp_image_file = "temp.png"
temp_folder = tempfile.gettempdir()

temp_image_filepath = os.path.join(temp_folder, temp_image_file)

worker = ud.UD_Processor()

class Run_UD(Operator):
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
    
class Unload_UD(Operator):
    bl_idname = "image.unload_ud"
    bl_label = "Release memory"

    def execute(self, context):
        worker.unload()
        return {'FINISHED'}
    
class Controlnet_AddItem(Operator):
    bl_idname = "controlnet.add_item"
    bl_label = "Add ControlNet Item"

    def execute(self, context):
        ws = context.workspace  # Replace with your actual data path
        ws.ud.controlnet_list.add()  # Adjust this line based on how you access your list

        return {'FINISHED'}
    
class Controlnet_RemoveItem(Operator):
    bl_idname = "controlnet.remove_item"
    bl_label = "Remove Controlnet"

    item_index: bpy.props.IntProperty()  

    def execute(self, context):
        ws = context.workspace
        ws.ud.controlnet_list.remove(self.item_index)
        
        return {'FINISHED'}
    
class Generate_ZDepthMap(Operator):
    bl_idname = "generate.zdepthmap"
    bl_label = "Generate Z Depth Map"

    def execute(self, context):
        context = bpy.context
        ws = bpy.context.workspace

        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                image_area = area

        # Save Settings
        settings_to_save = [
            ('context.scene', 'camera'),
            ('context.scene.render', 'engine'),
            ('context.view_layer', 'use_pass_z'),
            ('context.scene.eevee', 'taa_render_samples'),
            ('context.scene.render','resolution_x'),
            ('context.scene.render','resolution_y'),
            ('context.scene.render','resolution_percentage'),
        ]
        
        saved_settings = {}
        for (obj_path, attr) in settings_to_save:
            saved_settings[obj_path+'.'+attr] = getattr(eval(f"bpy.{obj_path}"), attr)

        # Create a new camera
        bpy.ops.object.camera_add()
        temp_camera = bpy.context.object

        # Align the new camera to the current view
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                rv3d = area.spaces[0].region_3d

                vmat_inv = rv3d.view_matrix.inverted()
                pmat = rv3d.perspective_matrix @ vmat_inv
                fov = 2.0*math.atan(1.0/pmat[1][1])

                temp_camera.location = rv3d.view_matrix.inverted().translation
                temp_camera.rotation_euler = rv3d.view_rotation.to_euler()
                temp_camera.data.angle = fov
                context.scene.render.resolution_x = ws.ud.width
                context.scene.render.resolution_y = ws.ud.height
                context.scene.render.resolution_percentage = ws.ud.scale
                break

        # Set the new settings
        context.scene.camera = temp_camera
        context.scene.render.engine = 'BLENDER_EEVEE'
        context.view_layer.use_pass_z = True
        context.scene.eevee.taa_render_samples = 1

        # Create input render layer node
        context.scene.use_nodes = True
        tree = context.scene.node_tree
        links = tree.links

        layers = tree.nodes.new('CompositorNodeRLayers')
        layers.layer = context.window.view_layer.name

        # Create normalize node
        normalize = tree.nodes.new(type="CompositorNodeNormalize")

        # Create invert node
        invert_node = tree.nodes.new(type='CompositorNodeInvert')

        # Color Space (not used, the current implementation does not visibily improve quality)
        # color_space_node = tree.nodes.new(type='CompositorNodeConvertColorSpace')  # Color Space node
        # color_space_node.from_color_space = 'AgX Log'  # Set input color space (if needed)
        # color_space_node.to_color_space = 'Non-Color'  # Set output color space 

        # Create File Output node
        file_out = tree.nodes.new(type="CompositorNodeViewer")
        tree.nodes.active = file_out
        
        links.new(layers.outputs['Depth'], normalize.inputs[0])
        links.new(normalize.outputs[0], invert_node.inputs[1])
        links.new(invert_node.outputs[0], file_out.inputs[0])

        # # Render the scene
        bpy.ops.render.render(layer="ViewLayer", write_still=True)

        # # Save image
        image = bpy.data.images['Viewer Node']
        temp_filepath = temp_image_filepath
        image.save_render(filepath=temp_filepath)
        
        # # Clean up
        bpy.data.objects.remove(temp_camera)
        for node in [layers, normalize, invert_node, file_out]:
            tree.nodes.remove(node)

        # Load the image in the depth slot
        image = bpy.data.images.load(temp_image_filepath)
        image.name = 'depth'
        image_area.spaces.active.image = image

        if 'depth.001' in bpy.data.images:
            bpy.data.images.remove(bpy.data.images['depth.001'])

        # Restore Settings
        for (obj_path, attr) in settings_to_save:
            obj = eval(f"bpy.{obj_path}")
            setattr(obj, attr, saved_settings[obj_path+'.'+attr])
            
        return {'FINISHED'}