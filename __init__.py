"""AI Texture Projector: Properties editor > Material tab > AI Texture.

File descriptions:

    properties.py  settings (scene) and server details (add-on preferences)
    ui.py          the panels you see
    operators.py   what the buttons do
    capture.py     renders the object from six sides into one cross image
    client.py      talks HTTP to the AI server (runs on a background thread)
    jobs.py        watches the background request and applies the result
    texturing.py   writes the UV map and material onto the mesh
    projection.py  the pure maths shared by capture and texturing
"""

# When you run "Reload Scripts", Blender re-imports this file but would keep
# the old copies of the other modules. Reloading them here picks up edits.
if "bpy" in locals():
    import importlib
    for _mod in (projection, client, texturing, capture, jobs, properties, operators, ui):
        importlib.reload(_mod)
else:
    from . import projection, client, texturing, capture, jobs, properties, operators, ui

import bpy

classes = (*properties.classes, *operators.classes, *ui.classes)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    properties.register()


def unregister():
    jobs.shutdown()
    properties.unregister()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
