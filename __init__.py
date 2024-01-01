import bpy, sys, os

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

def register():
    bpy.utils.register_class(op.Run_UD)
    bpy.utils.register_class(op.Unload_UD)

    bpy.utils.register_class(pg.UDPropertyGroup)
    bpy.utils.register_class(pn.UDPanel)    

    bpy.types.WorkSpace.ud = bpy.props.PointerProperty(type=pg.UDPropertyGroup)

def unregister():
    bpy.utils.unregister_class(op.Run_UD)
    bpy.utils.unregister_class(op.Unload_UD)

    bpy.utils.unregister_class(pg.UDPropertyGroup)
    bpy.utils.unregister_class(pn.UDPanel)

if __name__ == "__main__":
    register()