"""Buttons: Generate, Cancel, Preview Capture, and manual Project Cubemap."""

import bpy

from . import capture, client, jobs, projection, providers, texturing
from .properties import get_prefs, provider_key, provider_problem


def _capture(context, obj, s):
    wm = context.window_manager
    wm.progress_begin(0, len(projection.SIDES))
    try:
        return capture.capture_views(obj, int(s.face_size), s.front_axis, s.uniform,
                                     s.capture_style, progress=wm.progress_update)
    finally:
        wm.progress_end()


def _active_mesh(context):
    obj = context.object
    return obj if obj is not None and obj.type == "MESH" else None


class AITEX_OT_generate(bpy.types.Operator):
    """Capture the active object from six sides, send the views and your prompt to
    the AI server, and project the returned texture back onto the object"""
    bl_idname = "ai_texture.generate"
    bl_label = "Generate Texture"

    @classmethod
    def poll(cls, context):
        prefs = get_prefs(context)
        if _active_mesh(context) is None:
            cls.poll_message_set("Select a mesh object")
            return False
        if jobs.is_running():
            cls.poll_message_set("A request is already running")
            return False
        if not context.scene.ai_texture.prompt.strip():
            cls.poll_message_set("Type a prompt first")
            return False
        problem = provider_problem(prefs)
        if problem:
            cls.poll_message_set(problem)
            return False
        return True

    def execute(self, context):
        s = context.scene.ai_texture
        prefs = get_prefs(context)
        obj = _active_mesh(context)
        provider = prefs.provider
        hosted = provider != "CUSTOM"
        url = providers.OPENAI_EDIT_URL if provider == "OPENAI" else \
            providers.GEMINI_URL if provider == "GEMINI" else prefs.server_url.strip()

        problem = provider_problem(prefs)
        if problem:
            self.report({"ERROR"}, problem)
            return {"CANCELLED"}
        # Blender 4.2+: respect Preferences > System > Network > Allow Online Access.
        if not getattr(bpy.app, "online_access", True) and not client.is_local(url):
            self.report({"ERROR"}, "Online access is turned off. Enable 'Allow Online "
                                   "Access' in Preferences > System > Network")
            return {"CANCELLED"}
        if not obj.data.polygons:
            self.report({"ERROR"}, f"'{obj.name}' has no faces to texture")
            return {"CANCELLED"}

        try:
            image, png, bounds = _capture(context, obj, s)
        except Exception as err:
            self.report({"ERROR"}, f"Capturing the views failed: {err}")
            return {"CANCELLED"}
        s.last_capture = image
        width, height = image.size

        job = jobs.Job(obj.name, context.scene.name, bounds, {
            "front_axis": s.front_axis, "uniform": s.uniform,
            "padding": s.padding, "shading": s.shading,
            # Hosted models return their own resolution; stretch it back to the
            # capture's exact size so it lines up with the views we sent.
            "fit_to": (width, height) if hosted else None,
        }, provider_label=providers.PROVIDERS[provider]["label"] if hosted else "")

        # Start the worker to send the job to the server for processing.
        context_prompt = "You are given six sides of an object. the top side is on the top, the bottom side is on the bottom. Th left, right, forward and backward sides are on the middle. You have to add color to the object. Do not add any extra objects, only paint on top of the given object. "
        if hosted:
            context_prompt += providers.ALIGNMENT_PROMPT
        context_prompt += s.prompt.strip()

        request = {
            "provider": provider,
            "timeout": prefs.timeout,
            "prompt": context_prompt,
            "face_size": int(s.face_size), "faces": capture.faces_description(),
        }
        if hosted:
            # Hosted models handle transparency poorly: send a plain background.
            request.update({
                "image_png": capture.flatten_png(image),
                "api_key": provider_key(prefs),
                "model": (prefs.openai_model if provider == "OPENAI" else prefs.gemini_model).strip(),
                "quality": prefs.openai_quality,
                "size": (width, height),
            })
        else:
            request.update({
                "image_png": png, "server_url": url, "api_key": prefs.api_key,
                "poll_interval": prefs.poll_interval,
                "max_run_time": prefs.max_job_minutes * 60,
            })
        jobs.start(job, request)
        where = providers.PROVIDERS[provider]["label"] if hosted else "the AI server"
        self.report({"INFO"}, f"Sent '{obj.name}' to {where}")
        return {"FINISHED"}


class AITEX_OT_cancel(bpy.types.Operator):
    """Stop waiting for the server (the server may still finish its work)"""
    bl_idname = "ai_texture.cancel"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, context):
        return jobs.is_running()

    def execute(self, context):
        jobs.cancel()
        return {"FINISHED"}


class AITEX_OT_preview_capture(bpy.types.Operator):
    """Make the six-view image without sending it, to check what the server will see.
    Open it in an Image Editor to inspect or save it"""
    bl_idname = "ai_texture.preview_capture"
    bl_label = "Preview Capture"

    @classmethod
    def poll(cls, context):
        if _active_mesh(context) is None:
            cls.poll_message_set("Select a mesh object")
            return False
        return True

    def execute(self, context):
        s = context.scene.ai_texture
        try:
            image, _png, _bounds = _capture(context, _active_mesh(context), s)
        except Exception as err:
            self.report({"ERROR"}, f"Capturing the views failed: {err}")
            return {"CANCELLED"}
        s.last_capture = image
        self.report({"INFO"}, f"Captured '{image.name}' ({image.size[0]}×{image.size[1]})")
        return {"FINISHED"}


class AITEX_OT_project_cubemap(bpy.types.Operator):
    """Project the chosen cubemap image onto the selected mesh objects from all six sides"""
    bl_idname = "ai_texture.project_cubemap"
    bl_label = "Project Cubemap"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.scene.ai_texture.image is None:
            cls.poll_message_set("Choose a cubemap image first")
            return False
        if not any(ob.type == "MESH" for ob in context.selected_objects):
            cls.poll_message_set("Select at least one mesh object")
            return False
        return True

    def execute(self, context):
        s = context.scene.ai_texture
        image = s.image
        width, height = image.size
        if not width or not height:
            self.report({"ERROR"}, f"Couldn't read '{image.name}'. Is the file missing?")
            return {"CANCELLED"}

        layout = s.layout
        if layout == "AUTO":
            layout = projection.detect_layout(width, height)
            if layout is None:
                if projection.looks_equirectangular(width, height):
                    msg = ("This is a 2:1 panorama (equirectangular), not a cubemap. "
                           "Convert it to a cubemap first")
                else:
                    msg = (f"Can't tell the layout of a {width}×{height} image. "
                           "Pick one under Layout")
                self.report({"ERROR"}, msg)
                return {"CANCELLED"}

        cols, rows, _cells = projection.LAYOUTS[layout]
        if abs(width / cols - height / rows) > 1.0:
            self.report({"WARNING"},
                        f"Faces aren't square for {projection.LAYOUT_NAMES[layout]} "
                        f"({width / cols:.0f}×{height / rows:.0f} px). Check the layout")

        material = texturing.ensure_material(image, s.shading, key=image,
                                             name=f"Cubemap {image.name}")
        rotations = {side: int(getattr(s, f"turn_{side.lower()}")) for side in projection.SIDES}
        flips = {side: getattr(s, f"flip_{side.lower()}") for side in projection.SIDES}

        # Mesh data can only be written reliably in Object Mode.
        was_editing = context.mode == "EDIT_MESH"
        if was_editing:
            bpy.ops.object.mode_set(mode="OBJECT")

        done, seen = 0, set()
        for obj in context.selected_objects:
            if obj.type != "MESH" or obj.data in seen:
                continue
            seen.add(obj.data)
            error = texturing.apply_cubemap(
                obj, image, material, layout=layout, mode=s.mode,
                front_axis=s.front_axis, uniform=s.uniform, padding=s.padding,
                rotations=rotations, flips=flips, swap_front_back=s.swap_front_back)
            if error:
                self.report({"WARNING"}, error)
            else:
                done += 1

        if was_editing:
            bpy.ops.object.mode_set(mode="EDIT")
        self.report({"INFO"}, f"Projected {projection.LAYOUT_NAMES[layout]} onto "
                              f"{done} object{'s' if done != 1 else ''}")
        return {"FINISHED"}


classes = (AITEX_OT_generate, AITEX_OT_cancel, AITEX_OT_preview_capture,
           AITEX_OT_project_cubemap)
