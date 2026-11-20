"""Settings: AI provider, keys and server details in the add-on preferences
(shared by every .blend file), everything else on the scene (saved with the .blend)."""

import math

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from . import client, providers

ADDON_ID = __package__

PROVIDER_ITEMS = [
    ("CUSTOM", "Custom Server", "Your own server (see README.md for the API)", "URL", 0),
    ("OPENAI", "ChatGPT", "OpenAI's GPT Image models, with your OpenAI API key", "SHADERFX", 1),
    ("GEMINI", "Nano Banana", "Google's Gemini image models, with your Gemini API key",
     "SHADERFX", 2),
]


def get_prefs(context=None):
    addon = (context or bpy.context).preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


def provider_key(prefs, provider=None):
    """The API key for a hosted provider: the preference, else an environment
    variable (OPENAI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY)."""
    provider = provider or prefs.provider
    typed = {"OPENAI": prefs.openai_api_key, "GEMINI": prefs.gemini_api_key}.get(provider, "")
    return typed.strip() or providers.key_from_env(provider)


def provider_problem(prefs):
    """Why Generate can't run with the current provider settings, or None."""
    if prefs is None:
        return "Add-on preferences not found"
    if prefs.provider == "CUSTOM":
        url = prefs.server_url.strip()
        if not url:
            return "Set the server URL in the add-on preferences"
        return client.check_url(url)
    label = providers.PROVIDERS[prefs.provider]["label"]
    if not provider_key(prefs):
        return f"Add your {label} API key in the add-on preferences"
    model = prefs.openai_model if prefs.provider == "OPENAI" else prefs.gemini_model
    if not model.strip():
        return f"Set the {label} model in the add-on preferences"
    return None


class AITexPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    provider: EnumProperty(
        name="AI Provider",
        description="Where the six views are sent to be textured",
        items=PROVIDER_ITEMS,
        default="CUSTOM",
    )

    # --- ChatGPT (OpenAI) ------------------------------------------------------
    openai_api_key: StringProperty(
        name="OpenAI API Key",
        description="From platform.openai.com/api-keys. If empty, the OPENAI_API_KEY "
                    "environment variable is used",
        subtype="PASSWORD",
    )
    openai_model: StringProperty(
        name="Model",
        description="GPT Image model. It must support custom sizes for the 4:3 layout, "
                    "e.g. gpt-image-2.5-sunburst (best for precise edits) or "
                    "gpt-image-2.5-flare (faster)",
        default=providers.PROVIDERS["OPENAI"]["default_model"],
    )
    openai_quality: EnumProperty(
        name="Quality",
        description="Higher quality costs more and takes longer",
        items=[("low", "Low", ""), ("medium", "Medium", ""), ("high", "High", ""),
               ("auto", "Auto", "Let OpenAI choose")],
        default="medium",
    )

    # --- Nano Banana (Google Gemini) --------------------------------------------
    gemini_api_key: StringProperty(
        name="Gemini API Key",
        description="From aistudio.google.com/apikey. If empty, the GEMINI_API_KEY "
                    "(or GOOGLE_API_KEY) environment variable is used",
        subtype="PASSWORD",
    )
    gemini_model: StringProperty(
        name="Model",
        description="Gemini image model, e.g. gemini-nano-banana-2.1 (Nano Banana 2.1) "
                    "or gemini-3-pro-image (Nano Banana Pro)",
        default=providers.PROVIDERS["GEMINI"]["default_model"],
    )

    # --- Custom server ---------------------------------------------------------
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
        name="Response Timeout",
        description="Give up if a request gets no answer for this many seconds: the "
                    "upload to a custom server, or the whole call to ChatGPT / Nano "
                    "Banana. A custom server that keeps reporting 'queued' or "
                    "'running' is waited for as long as it takes",
        default=300, min=10, max=3600,
    )
    max_job_minutes: IntProperty(
        name="Max Job Time",
        description="Stop waiting if a job has been running on the custom server for "
                    "longer than this many minutes. 0 = no limit (you can always "
                    "press Cancel)",
        default=0, min=0, max=24 * 60,
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
        layout.use_property_decorate = False
        layout.prop(self, "provider")

        box = layout.box()
        if self.provider == "CUSTOM":
            box.prop(self, "server_url")
            box.prop(self, "api_key")
            box.prop(self, "poll_interval", text="Check Every (s)")
            box.prop(self, "max_job_minutes", text="Max Job Time (min, 0 = no limit)")
            box.label(text="See README.md inside the add-on for the request/response format.",
                      icon="INFO")
        else:
            info = providers.PROVIDERS[self.provider]
            if self.provider == "OPENAI":
                box.prop(self, "openai_api_key", text="API Key")
                box.prop(self, "openai_model")
                box.prop(self, "openai_quality")
                typed = self.openai_api_key
            else:
                box.prop(self, "gemini_api_key", text="API Key")
                box.prop(self, "gemini_model")
                typed = self.gemini_api_key
            if not typed.strip() and providers.key_from_env(self.provider):
                box.label(text=f"Using the key from ${info['env'][0]}", icon="CHECKMARK")
            row = box.row()
            row.operator("wm.url_open", text="Get an API Key", icon="URL").url = info["key_page"]
            col = box.column(align=True)
            col.label(text=f"Each Generate sends the six views and your prompt to "
                           f"{info['long']}", icon="INFO")
            col.label(text="and is billed to your API key.", icon="BLANK1")

        layout.prop(self, "timeout", text="Response Timeout (s)")
        layout.label(text="API keys are stored unencrypted in Blender's preferences.",
                     icon="LOCKED")


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
        description="Don't stretch the sides into square cells. AI capture: the "
                    "object's box is unfolded at its true proportions, so wide or "
                    "tall objects fill the image tightly (the image is then not 4:3). "
                    "Existing cubemaps: the same scale on every axis, so shorter "
                    "sides use only the middle part of their square",
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

    # --- How AI results are put on the object --------------------------------
    ai_projection: EnumProperty(
        name="Method",
        description="How the painted views are put onto the object",
        items=[
            ("BAKE", "Blend & Bake", "Blend all views into one UV texture, favouring the view "
                                     "that sees each spot most directly. Soft seams, no "
                                     "stretching on slanted faces, hidden areas filled in"),
            ("BOX", "Box", "Each face takes the view it points at most (the original method). "
                           "Fast, but hard seams and stretching on slanted faces"),
        ],
        default="BAKE",
    )
    bake_size: EnumProperty(
        name="Texture Size",
        items=[("1024", "1024 px", ""), ("2048", "2048 px", ""), ("4096", "4096 px", "")],
        default="2048",
    )
    bake_uvs: EnumProperty(
        name="UVs",
        items=[
            ("AUTO", "Auto Unwrap", "Smart UV Project into a 'UVUSeeBake' UV map (your other "
                                    "UV maps are kept)"),
            ("ACTIVE", "Active UV Map", "Use your own active UV map. Overlapping islands "
                                        "(e.g. mirrored halves) will share colours"),
        ],
        default="AUTO",
    )
    bake_sharpness: FloatProperty(
        name="Blend Sharpness",
        description="How strongly the most direct view wins where views overlap. "
                    "Low: smoother blends; high: crisper detail, more visible seams",
        default=4.0, min=1.0, max=16.0,
    )
    bake_max_angle: FloatProperty(
        name="Max View Angle",
        description="Ignore a view where it sees the surface at a steeper angle than "
                    "this (stretched, unreliable pixels)",
        default=math.radians(80.0), min=math.radians(30.0), max=math.radians(89.0),
        subtype="ANGLE",
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