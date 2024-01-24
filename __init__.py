import bpy, sys, os
from bpy.utils import register_class, unregister_class

script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.join(script_dir, 'dependencies')
sys.path.append(venv_path)

import importlib
import inspect

from . import property_groups as pg
from . import operators as op
from . import panels as pn
from . import ud_processor as ud

classes_to_register = []
for module in [pg,op,pn]:
    importlib.reload(module)
    classes_in_module = [ cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__ ]
    classes_to_register.extend(classes_in_module)

for module in [ud]:
    importlib.reload(module)

bl_info = {
    "name" : "Blender Unexpected Diffusion",
    "author" : "Nicolas Predella",
    "description" : "",
    "blender" : (4, 0, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

def register():
    for class_to_register in classes_to_register:
        register_class(class_to_register)

    bpy.types.WorkSpace.ud = bpy.props.PointerProperty(type=pg.UDPropertyGroup)

def unregister():
    for class_to_register in classes_to_register:
        unregister_class(class_to_register)

if __name__ == "__main__":
    register()