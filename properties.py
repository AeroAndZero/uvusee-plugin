"""Settings: server details in the add-on preferences (shared by every
.blend file), everything else on the scene (saved with the .blend)."""

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

ADDON_ID = __package__


def get_prefs(context=None):
    addon = (context or bpy.context).preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


class AITexPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    server_url: StringProperty(
        name="Server URL",
        description="Address the captured views are POSTed to, "
                    "e.g. https://my-server.example.com/texture",
    )
    api_key: StringProperty(
        name="API Key",
        description="Sent as 'Authorization: Bearer <key>'. Leave empty if your "
                    "server doesn't need one",
        subtype="PASSWORD",
    )
    timeout: IntProperty(
        name="Timeout",
        description="Give up if a job runs on the server for longer than this many "
                    "seconds. Time spent waiting in the server's queue doesn't count",
        default=300, min=10, max=3600,
    )
    poll_interval: FloatProperty(
        name="Check Every",
        description="Seconds between status checks while a job runs. While it waits "
                    "in the queue with no change, checks slow down to 5× this",
        default=5.0, min=0.5, max=60.0,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "server_url")
        layout.prop(self, "api_key")
        layout.prop(self, "timeout", text="Timeout (s)")
        layout.prop(self, "poll_interval", text="Check Every (s)")
        col = layout.column(align=True)
        col.label(text="The API key is stored unencrypted in Blender's preferences.",
                  icon="INFO")
        col.label(text="See README.md inside the add-on for the request/response format.")


LAYOUT_ITEMS = [
    ("AUTO", "Auto Detect", "Guess the layout from the image's proportions"),
    ("CROSS_H", "Horizontal Cross", "4×3: top / left front right back / bottom"),
    ("CROSS_V", "Vertical Cross", "3×4: top / left front right / bottom / back (upside down)"),
    ("STRIP_H", "Horizontal Strip", "6×1 in +X −X +Y −Y +Z −Z order: right left top bottom front back"),
    ("STRIP_V", "Vertical Strip", "1×6 in +X −X +Y −Y +Z −Z order: right left top bottom front back"),
    ("GRID_3X2", "3×2 Grid", "Right left top / bottom front back"),
]

TURN_ITEMS = [
    ("0", "0°", "As stored"),
    ("1", "90°", "Rotate 90° counter-clockwise"),
    ("2", "180°", "Rotate 180°"),
    ("3", "270°", "Rotate 90° clockwise"),
]


def _turns(name):
    return EnumProperty(name=f"{name} Rotation", items=TURN_ITEMS, default="0")


def _flip(name):
    return BoolProperty(name=f"Mirror {name}", description="Mirror this face left-to-right")


class AITexSettings(bpy.types.PropertyGroup):
    # --- AI generation -------------------------------------------------------
    prompt: StringProperty(
        name="Prompt",
        description="Describe the texture you want, e.g. 'weathered oak treasure "
                    "chest with iron bands'",
    )
    face_size: EnumProperty(
        name="Capture Size",
        description="Resolution of each of the six views sent to the server",
        items=[("256", "256 px", ""), ("512", "512 px", ""),
               ("1024", "1024 px", ""), ("2048", "2048 px", "")],
        default="512",
    )
    capture_style: EnumProperty(
        name="Capture Style",
        items=[
            ("CLAY", "Clay", "Neutral grey with shading, shows only the shape"),
            ("TEXTURED", "Current Look", "The object's current colours and textures, "
                                         "e.g. to refine a previous result"),
        ],
        default="CLAY",
    )
    last_capture: PointerProperty(type=bpy.types.Image, name="Last Capture")
    last_result: PointerProperty(type=bpy.types.Image, name="Last Result")

    # --- Projection (used for AI results and manual cubemaps) ----------------
    front_axis: EnumProperty(
        name="Front",
        description="Which side of the object counts as its front",
        items=[
            ("NEG_Y", "−Y", "Blender's Front view side (default)"),
            ("POS_Y", "+Y", "Blender's Back view side"),
            ("POS_X", "+X", "Blender's Right view side"),
            ("NEG_X", "−X", "Blender's Left view side"),
        ],
        default="NEG_Y",
    )
    uniform: BoolProperty(
        name="Keep Proportions",
        description="Use the same scale on every axis so images aren't stretched on "
                    "non-cubic objects. Shorter sides then use only the middle "
                    "part of their image",
        default=False,
    )
    padding: FloatProperty(
        name="Edge Padding",
        description="Pixels to stay away from each face's border, so texture "
                    "filtering doesn't pick up the neighbouring face",
        default=1.0, min=0.0, max=16.0, subtype="PIXEL",
    )
    shading: EnumProperty(
        name="Shading",
        items=[
            ("PRINCIPLED", "Lit", "Principled BSDF: reacts to scene lighting"),
            ("EMISSION", "Flat", "Emission: shows the image exactly, ignoring lights"),
        ],
        default="PRINCIPLED",
    )

    # --- Manual cubemap --------------------------------------------------------
    image: PointerProperty(
        name="Cubemap",
        type=bpy.types.Image,
        description="Cubemap image containing all six faces",
    )
    layout: EnumProperty(name="Layout", items=LAYOUT_ITEMS, default="AUTO")
    mode: EnumProperty(
        name="Projection",
        items=[
            ("BOX", "Box", "Project straight onto each side, chosen by which way each face "
                           "points. Best for boxy and general shapes"),
            ("SPHERE", "Sphere", "Project through the object's centre like a real cubemap "
                                 "lookup. Seamless on round objects"),
        ],
        default="BOX",
    )
    swap_front_back: BoolProperty(
        name="Swap Front/Back",
        description="Some formats treat +Z as the back rather than the front",
    )
    turn_front: _turns("Front")
    turn_back: _turns("Back")
    turn_left: _turns("Left")
    turn_right: _turns("Right")
    turn_top: _turns("Top")
    turn_bottom: _turns("Bottom")
    flip_front: _flip("Front")
    flip_back: _flip("Back")
    flip_left: _flip("Left")
    flip_right: _flip("Right")
    flip_top: _flip("Top")
    flip_bottom: _flip("Bottom")


classes = (AITexPreferences, AITexSettings)


def register():
    bpy.types.Scene.ai_texture = PointerProperty(type=AITexSettings)


def unregister():
    del bpy.types.Scene.ai_texture
