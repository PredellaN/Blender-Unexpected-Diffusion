import bpy, os, tempfile, threading, random, math
from bpy.types import Operator
from . import PG_NAME_LC, blender_globals
from . import property_groups as pg
from .functions import ud_classes as udcl
from .constants import DIFFUSION_MODELS

worker = None

# Create a temporary file path that works on both Windows and Unix-like systems
temp_image_file = "temp.png"
temp_folder = tempfile.gettempdir()

temp_image_filepath = os.path.join(temp_folder, temp_image_file)

def get_model_type(model_id):
    for model in DIFFUSION_MODELS:
        if model.id == model_id:
            return model.type
    return None  # Return None if the ID is not found

class Run_UD(Operator):
    bl_idname = f"{PG_NAME_LC}.run_ud"
    bl_label = "Run Unexpected Diffusion"

    mode: bpy.props.StringProperty() # type: ignore

    def ud_task(self, params, image_area, manager):
        from . import ud_processor as ud
        global worker
        worker = ud.UD_Processor()

        try:
            result = worker.run(params=params, manager=manager)
            if result:
                image = bpy.data.images.load(params['temp_image_filepath'])
                image.name = params['prompt'][:57] + "-" + str(params['seed'])
                image_area.spaces.active.image = image

        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            manager.set_running(0)

    def ud_upscale_task(self, params, image_area, manager):
        from . import ud_processor as ud
        global worker
        worker = ud.UD_Processor()

        try:
            space = image_area.spaces.active

            if space.image:
                params['width'] = space.image.size[0]
                params['height'] = space.image.size[1]
            
                original_view_transform = bpy.context.scene.view_settings.view_transform
                bpy.context.scene.view_settings.view_transform = 'Raw'
                bpy.data.images[space.image.name].save_render(params['temp_image_filepath'])
                bpy.context.scene.view_settings.view_transform = original_view_transform

                worker.upscale(params=params, manager=manager)

                image = bpy.data.images.load(params['temp_image_filepath'])
                image.name = params['prompt'][:57] + "-" + str(params['seed'])
                image_area.spaces.active.image = image
                
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            manager.set_running(0)

    def execute(self, context):
        areas = bpy.context.screen.areas
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)

        pg.running = 1
        pg.progress = 0
        pg.progress_text = ""

        # Prepare params
        params = {prop.identifier: getattr(pg, prop.identifier) 
                   for prop in pg.bl_rna.properties 
                   if not prop.is_readonly}

        # Prepare manager
        manager = udcl.ProcessManager(ws, pg)

        # Programmatic params
        params['temp_image_filepath'] = temp_image_filepath
        params['pipeline_type'] = get_model_type(params['model'])

        if pg.seed == 0:
            params['seed'] = random.randint(1, 99999)

        cm = pg.control_mode
        for item in getattr(pg, f'{cm}_list'):
            if getattr(item, f'{cm}_image_slot') and getattr(item, f'{cm}_factor') > 0:
                for entry in [f'{cm}_model',f'{cm}_image_slot',f'{cm}_factor']:
                    if not params.get(entry):
                        params[entry]=[]
                    params[entry].append(getattr(item, entry))

        params['mode'] = self.mode
        print(params)
        
        for area in areas:
            if area.type == 'IMAGE_EDITOR':
                image_area = area

        if self.mode in ['generate']: 
            thread = threading.Thread(target=self.ud_task, args=[params, image_area, manager])
        elif self.mode in ['upscale_sd','upscale_re']:
            thread = threading.Thread(target=self.ud_upscale_task, args=[params, image_area, manager])
        
        thread.start()

        return {'FINISHED'}
    
class Unload_UD(Operator):
    bl_idname = f"{PG_NAME_LC}.unload_ud"
    bl_label = "Release memory"

    def execute(self, context):
        global worker
        if worker:
            worker.unload()
        return {'FINISHED'}
    
class Stop_UD(Operator):
    bl_idname = f"{PG_NAME_LC}.stop_ud"
    bl_label = "Stop generation"

    def execute(self, context):
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)
        pg.stop_process = 1
        return {'FINISHED'}
    
class Control_Mode(Operator):
    bl_idname = f"{PG_NAME_LC}.control_mode"
    bl_label = "Set Control Mode"

    switch_mode: bpy.props.StringProperty() # type: ignore

    def execute(self, context):
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)
        pg.control_mode = self.switch_mode
        return {'FINISHED'}
    
class Control_AddItem(Operator):
    bl_idname = f"{PG_NAME_LC}.control_add_item"
    bl_label = "Add ControlNet Item"

    def execute(self, context):
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)
        control_list = getattr(pg, f'{pg.control_mode}_list')
        control_list.add()
        return {'FINISHED'}
    
class Control_RemoveItem(Operator):
    bl_idname = f"{PG_NAME_LC}.control_remove_item"
    bl_label = "Remove Controlnet"

    item_index: bpy.props.IntProperty() # type: ignore

    def execute(self, context):
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)
        control_list = getattr(pg, f'{pg.control_mode}_list')
        control_list.remove(self.item_index)
        return {'FINISHED'}
    
class Project_UVs(bpy.types.Operator):
    bl_idname = f"{PG_NAME_LC}.generate_projected_uvs"
    bl_label = "Project UVs from View"

    def execute(self, context):
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                
                if context.active_object.mode != 'EDIT':
                    self.report({'WARNING'}, "Operation requires Edit Mode")
                    return {'CANCELLED'}
                
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with context.temp_override(area=area, region=region):
                            bpy.ops.uv.project_from_view(scale_to_bounds=False) #project

                break
        else:
            self.report({'WARNING'}, "Operation requires one active 3d View area")
            return {'CANCELLED'}

        bpy.ops.uv.select_all(action='SELECT')
        bpy.ops.transform.resize(
            value=(1, pg.width / pg.height, 1), orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)))

        return {'FINISHED'}

class Generate_Map(Operator):
    bl_idname = f"{PG_NAME_LC}.generate_map"
    bl_label = "Generate Map"

    mode: bpy.props.StringProperty() # type: ignore
    target: bpy.props.StringProperty() # type: ignore

    def execute(self, context):
        ws = context.workspace
        pg = getattr(ws, PG_NAME_LC)

        if self.target == '3d':
            # # Save original status
            original_selection = context.selected_objects.copy()
            original_active = context.view_layer.objects.active
            original_mode = context.object.mode if bpy.context.active_object else None
        
            if original_mode:
                bpy.ops.object.mode_set(mode='OBJECT')

            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    image_area = area

            # Save Settings
            settings_to_save = [
                ('context.scene', 'camera'),
                ('context.scene.render', 'engine'),
                ('context.view_layer', 'use_pass_z'),
                ('context.view_layer', 'use_pass_normal'),
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
                    context.scene.render.resolution_x = pg.width
                    context.scene.render.resolution_y = pg.height
                    context.scene.render.resolution_percentage = pg.scale
                    break

            # Set the new settings
            context.scene.camera = temp_camera
            context.scene.render.engine = 'BLENDER_EEVEE'
            context.view_layer.use_pass_z = True
            context.view_layer.use_pass_normal = True
            context.scene.eevee.taa_render_samples = 1

            # # Setup Compositor
            # Create input render layer node
            context.scene.use_nodes = True
            tree = context.scene.node_tree
            links = tree.links
            node_setup = {}

            # Create Render Layer node
            node_setup['layers'] = tree.nodes.new('CompositorNodeRLayers')
            node_setup['layers'].layer = context.window.view_layer.name

            # Create File Output node
            node_setup['file_out'] = tree.nodes.new(type="CompositorNodeViewer")
            tree.nodes.active = node_setup['file_out']


            if self.mode in ['depth', 'canny']:
                node_setup['normalize'] = tree.nodes.new(type="CompositorNodeNormalize")
                node_setup['invert_node'] = tree.nodes.new(type='CompositorNodeInvert')

                links.new(node_setup['layers'].outputs['Depth'], node_setup['normalize'].inputs[0])
                links.new(node_setup['normalize'].outputs[0], node_setup['invert_node'].inputs[1])


            if self.mode in ['depth']:
                links.new(node_setup['invert_node'].outputs[0], node_setup['file_out'].inputs[0])

            elif self.mode in ['canny']:
                node_setup['separatexyz'] = tree.nodes.new(type="CompositorNodeSeparateXYZ")
                node_setup['combinexyz'] = tree.nodes.new(type="CompositorNodeCombineXYZ")

                links.new(node_setup['layers'].outputs['Normal'], node_setup['separatexyz'].inputs[0])
                
                for key, axis in enumerate(['x', 'y', 'z']):
                    node_setup[f'sum_{axis}'] = tree.nodes.new(type="CompositorNodeMath")
                    node_setup[f'sum_{axis}'].operation = 'ADD'
                    node_setup[f'sum_{axis}'].inputs[1].default_value = 1

                    node_setup[f'divide_{axis}'] = tree.nodes.new(type="CompositorNodeMath")
                    node_setup[f'divide_{axis}'].operation = 'DIVIDE'
                    node_setup[f'divide_{axis}'].inputs[1].default_value = 2

                    links.new(node_setup['separatexyz'].outputs[key], node_setup[f'sum_{axis}'].inputs[0])
                    links.new(node_setup[f'sum_{axis}'].outputs[0], node_setup[f'divide_{axis}'].inputs[0])
                    links.new(node_setup[f'divide_{axis}'].outputs[0], node_setup['combinexyz'].inputs[key])

                node_setup['separate_color'] = tree.nodes.new(type="CompositorNodeSeparateColor")
                node_setup['separate_color'].mode = 'HSV'
                links.new(node_setup['combinexyz'].outputs[0], node_setup['separate_color'].inputs[0])

                node_setup['combine_color'] = tree.nodes.new(type="CompositorNodeCombineColor")
                node_setup['combine_color'].mode = 'HSV'
                links.new(node_setup['separate_color'].outputs[0], node_setup['combine_color'].inputs[0])
                links.new(node_setup['separate_color'].outputs[1], node_setup['combine_color'].inputs[1])
                links.new(node_setup['invert_node'].outputs[0], node_setup['combine_color'].inputs[2])

                links.new(node_setup['combine_color'].outputs[0], node_setup['file_out'].inputs[0])

            # # Render the scene
            bpy.ops.render.render(layer="ViewLayer", write_still=True)

            # # Save image
            image = bpy.data.images['Viewer Node']
            temp_filepath = temp_image_filepath
            image.save_render(filepath=temp_filepath)

        elif self.target == 'image':
            areas = bpy.context.screen.areas
            for area in areas:
                if area.type == 'IMAGE_EDITOR':
                    image_area = area
                    break
        
            if not image_area:
                self.report({'WARNING'}, "No image is open")
                return {'CANCELLED'}
                
            space = area.spaces.active

            if space.image is None:
                self.report({'WARNING'}, "No image is open")
                return {'CANCELLED'}
            
            bpy.data.images[space.image.name].save_render(temp_image_filepath)

        # Out-of-blender processing
        if self.mode in ['canny']:
            import cv2

            image = cv2.imread(temp_image_filepath)

            edges = cv2.Canny(image, 600 * (1-pg.canny_strength), 1200 * (1-pg.canny_strength))
            cv2.imwrite(temp_image_filepath, edges)

        # Load the image in the depth slot
        slot_name = self.mode
        if slot_name in bpy.data.images:
            bpy.data.images[slot_name].filepath = temp_image_filepath
            bpy.data.images[slot_name].reload()
        else:
            image = bpy.data.images.load(temp_image_filepath)
            image.name = slot_name

        image_area.spaces.active.image = bpy.data.images[slot_name]


        # # Clean up
        if self.target == '3d':
            bpy.data.objects.remove(temp_camera)
            for key, node in node_setup.items():
                tree.nodes.remove(node)

            # Restore Settings
            for (obj_path, attr) in settings_to_save:
                obj = eval(f"bpy.{obj_path}")
                setattr(obj, attr, saved_settings[obj_path+'.'+attr])

            # Clear current selection
            for obj in context.selected_objects:
                obj.select_set(False)

            # Select originally selected objects
            for obj in original_selection:
                obj.select_set(True)

            # Set original active object
            context.view_layer.objects.active = original_active

            # Return to original mode if needed
            if original_mode and original_active:
                bpy.ops.object.mode_set(mode=original_mode)
            

        return {'FINISHED'}