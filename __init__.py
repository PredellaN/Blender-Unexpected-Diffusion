import bpy, sys, os
from bpy.utils import register_class, unregister_class

script_dir = os.path.dirname(os.path.abspath(__file__))
venv_path = os.path.join(script_dir, 'dependencies')
sys.path.append(venv_path)

import importlib

from . import panels as pn
from . import property_groups as pg
from . import operators as op
from . import ud_processor as ud

if "bpy" in locals():
    for module in [pn,pg,op,ud]:
        importlib.reload(module)

bl_info = {
    "name" : "Blender Unexpected Diffusion",
    "author" : "Nicolas Predella",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

classes_to_register = [
    op.Run_UD,
    op.Unload_UD,
    op.Controlnet_AddItem,
    op.Controlnet_RemoveItem,
    pg.ControlNetListItem,
    pg.MY_UL_ControlNetList,
    pg.UDPropertyGroup,
    pn.UDPanel,
]

def register():
    for class_to_register in classes_to_register:
        register_class(class_to_register)

    bpy.types.WorkSpace.ud = bpy.props.PointerProperty(type=pg.UDPropertyGroup)

def unregister():
    for class_to_register in classes_to_register:
        unregister_class(class_to_register)

if __name__ == "__main__":
    register()