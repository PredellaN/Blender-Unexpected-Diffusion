bl_info = {
    "name" : "Blender Unexpected Diffusion",
    "author" : "Nicolas Predella",
    "description" : "",
    "blender" : (4, 0, 0),
    "version" : (0, 0, 2),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

import bpy, sys, os, importlib, threading, subprocess, inspect, time, pkgutil
from bpy.utils import register_class, unregister_class

from .constants import DEPENDENCIES, DEPENDENCIES_DIR
sys.path.append(DEPENDENCIES_DIR)

## PREFERENCES
class AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    running: bpy.props.BoolProperty(default=False)

    def draw(self, context):
        layout = self.layout
        row = layout.row()

        if are_dependencies_installed(DEPENDENCIES, DEPENDENCIES_DIR):
            row.label(text="Dependencies installed", icon='CHECKMARK')
        else:
            row.operator("addon.install_dependencies")
            row.enabled = not self.running



class InstallDependenciesOperator(bpy.types.Operator):
    bl_idname = "addon.install_dependencies"
    bl_label = "Install Dependencies"

    def install_packages(self, packages, target_dir):
        prefs = bpy.context.preferences.addons[__package__].preferences 
        try:
            # subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-t', target_dir] + packages)
            time.sleep(2)
            unregister()
            register()
        except Exception as e:
            print(f"Installation failed: {str(e)}")
        finally:
            prefs.running = False

    def execute(self, context):
        prefs = bpy.context.preferences.addons[__package__].preferences 
        prefs.running = True

        thread = threading.Thread(target=self.install_packages, args=[DEPENDENCIES, DEPENDENCIES_DIR])
        thread.start()

        return {'FINISHED'}

def get_classes(modules):
    classes = []
    for module in modules:
        classes_in_module = [cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__]
        classes.extend(classes_in_module)
    return classes   

def are_dependencies_installed(dependencies, dependencies_dir):
    # Check if each dependency has a corresponding folder in the dependencies directory
    for dep in [dep.replace('-', '_') for dep in dependencies]:
        # Check if there's any folder that starts with the dependency name
        if not any(os.path.isdir(os.path.join(dependencies_dir, f)) and f.startswith(dep) for f in os.listdir(dependencies_dir)):
            return False
    return True

## REGISTERING
def register():
    global registered_classes
    registered_classes = []

    classes_to_register = [AddonPreferences, InstallDependenciesOperator]

    deps_installed = are_dependencies_installed(DEPENDENCIES, DEPENDENCIES_DIR)

    if deps_installed:
        from . import property_groups as pg, operators as op, panels as pn
        classes_to_register.extend(get_classes([pg, op, pn]))

    for cls in classes_to_register:
        register_class(cls)

    registered_classes.extend(classes_to_register)

    if deps_installed:
        bpy.types.WorkSpace.ud = bpy.props.PointerProperty(type=pg.UDPropertyGroup)

def unregister():
    global registered_classes

    for cls in registered_classes[::-1]:
        unregister_class(cls)

    registered_classes = []

if __name__ == "__main__":
    register()