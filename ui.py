"""Panels in the Properties editor, Material tab: "AI Texture" and its sub-sections.

Blender doesn't let add-ons create a new tab in the Properties editor, so the
add-on lives in the Material tab, the natural home for texturing. You can drag
the panel by its header to move it above the other material panels.
"""

import textwrap

import bpy

from . import jobs, projection
from .properties import ADDON_ID, get_prefs


class _AITexPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"


def _wrapped(layout, context, text, icon="NONE"):
    """Labels don't wrap in Blender, so split long messages ourselves."""
    width = max(20, int(context.region.width / 7.5)) if context.region else 50
    col = layout.column(align=True)
    for i, line in enumerate(textwrap.wrap(text, width) or [""]):
        col.label(text=line, icon=icon if i == 0 else "BLANK1")


def _thumbnail(layout, image, label):
    col = layout.column(align=True)
    col.label(text=label)
    if image is None:
        col.label(text="—")
        return
    try:
        image.preview_ensure()
        col.template_icon(icon_value=image.preview.icon_id, scale=5.0)
    except Exception:
        col.label(text=image.name, icon="IMAGE_DATA")


class AITEX_PT_main(_AITexPanel, bpy.types.Panel):
    bl_label = "AI Texture"
    bl_idname = "AITEX_PT_main"

    def draw_header(self, context):
        self.layout.label(icon="SHADERFX")

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_texture
        prefs = get_prefs(context)
        job = jobs.current()

        try:
            layout.prop(s, "prompt", text="", icon="TEXT", placeholder="Describe the texture…")
        except TypeError:                           # 'placeholder' needs a newer Blender
            layout.prop(s, "prompt", text="", icon="TEXT")

        if job and job.running:
            box = layout.box()
            _wrapped(box, context, job.message, icon="SORTTIME")
            text = f"{job.elapsed:.0f} s"
            if job.progress is not None and hasattr(box, "progress"):
                box.progress(factor=job.progress, text=f"{job.progress:.0%} · {text}")
            else:
                box.label(text=f"Elapsed: {text}")
            box.operator("ai_texture.cancel", icon="CANCEL")
        else:
            row = layout.row()
            row.scale_y = 1.6
            row.operator("ai_texture.generate", icon="SHADERFX")
            if job and job.state == "failed":
                _wrapped(layout.box(), context, job.message, icon="ERROR")
            elif job and job.state == "done":
                layout.label(text=job.message, icon="CHECKMARK")
            elif job and job.state == "cancelled":
                layout.label(text="Cancelled", icon="CANCEL")

        if s.last_capture or s.last_result:
            row = layout.row()
            _thumbnail(row, s.last_capture, "Sent")
            _thumbnail(row, s.last_result, "Received")

        row = layout.row(align=True)
        url = prefs.server_url.strip() if prefs else ""
        if url:
            row.label(text=url, icon="URL")
        else:
            row.label(text="No server set", icon="ERROR")
        op = row.operator("preferences.addon_show", text="", icon="PREFERENCES")
        op.module = ADDON_ID


class AITEX_PT_capture(_AITexPanel, bpy.types.Panel):
    bl_label = "Capture"
    bl_parent_id = "AITEX_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        s = context.scene.ai_texture
        layout.prop(s, "face_size")
        layout.prop(s, "capture_style", expand=True)
        layout.operator("ai_texture.preview_capture", icon="RESTRICT_RENDER_OFF")


class AITEX_PT_projection(_AITexPanel, bpy.types.Panel):
    bl_label = "Projection"
    bl_parent_id = "AITEX_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        s = context.scene.ai_texture
        layout.prop(s, "front_axis")
        layout.prop(s, "shading", expand=True)
        layout.prop(s, "padding")
        layout.prop(s, "uniform")


class AITEX_PT_manual(_AITexPanel, bpy.types.Panel):
    bl_label = "Use Existing Cubemap"
    bl_parent_id = "AITEX_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_texture
        layout.template_ID(s, "image", open="image.open")
        if s.image:
            width, height = s.image.size
            if not width:
                layout.label(text="Image can't be read", icon="ERROR")
            else:
                key = s.layout if s.layout != "AUTO" else projection.detect_layout(width, height)
                if key:
                    cols, _rows, _ = projection.LAYOUTS[key]
                    layout.label(text=f"{projection.LAYOUT_NAMES[key]}, "
                                      f"{width // cols} px faces", icon="CHECKMARK")
                elif projection.looks_equirectangular(width, height):
                    layout.label(text="2:1 panorama, not a cubemap", icon="ERROR")
                else:
                    layout.label(text=f"{width}×{height}: pick a layout", icon="ERROR")

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(s, "layout")
        col.prop(s, "mode", expand=True)
        layout.operator("ai_texture.project_cubemap", icon="MOD_UVPROJECT")


class AITEX_PT_faces(_AITexPanel, bpy.types.Panel):
    bl_label = "Face Adjustments"
    bl_parent_id = "AITEX_PT_manual"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_texture
        layout.label(text="Only if a face comes out rotated or mirrored.")
        layout.prop(s, "swap_front_back")
        col = layout.column(align=True)
        for side in projection.SIDES:
            key = side.lower()
            row = col.row(align=True)
            row.label(text=side.title())
            row.prop(s, f"turn_{key}", text="")
            row.prop(s, f"flip_{key}", text="", icon="MOD_MIRROR", toggle=True)


classes = (AITEX_PT_main, AITEX_PT_capture, AITEX_PT_projection, AITEX_PT_manual,
           AITEX_PT_faces)
