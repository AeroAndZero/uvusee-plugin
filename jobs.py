"""One AI request at a time: a background thread does the networking, and a
timer on Blender's main thread watches it and applies the result.

Why the split: waiting for a server can take minutes, and Blender's UI would
freeze if we waited in an operator. But Blender's data may only be changed
from the main thread, so the thread never touches bpy; it just fills in
fields on the Job, and the timer (which runs on the main thread) acts on them.
"""

import os
import tempfile
import threading
import time

import bpy

from . import client, projection, texturing

ACTIVE = {"uploading", "waiting", "received"}


class Job:
    def __init__(self, object_name, scene_name, bounds, options):
        self.object_name = object_name
        self.scene_name = scene_name
        self.bounds = bounds          # the capture's bounding box, reused when applying
        self.options = options        # projection settings at the time of sending
        self.state = "uploading"
        self.message = "Uploading…"
        self.progress = None
        self.result = None
        self.started = time.monotonic()
        self.finished = None
        self.cancel_event = threading.Event()

    @property
    def running(self):
        return self.state in ACTIVE

    @property
    def elapsed(self):
        return (self.finished or time.monotonic()) - self.started

    def finish(self, state, message):
        self.state = state
        self.message = message
        self.finished = time.monotonic()


_job = None


def current():
    return _job


def is_running():
    return _job is not None and _job.running


def start(job, request):
    """request: dict with server_url, api_key, timeout, poll_interval and the
    arguments of client.Client.request_texture."""
    global _job
    _job = job
    thread = threading.Thread(target=_worker, args=(job, request), daemon=True)
    thread.start()
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.25)


def cancel():
    if _job and _job.running:
        _job.cancel_event.set()
        _job.finish("cancelled", "Cancelled")
        _redraw()


def _worker(job, request):
    def on_status(message, progress=None):
        if job.running:
            job.state = "waiting"
            job.message = message
            job.progress = progress

    try:
        api = client.Client(request["server_url"], request["api_key"],
                            timeout=request["timeout"],
                            poll_interval=request["poll_interval"],
                            on_status=on_status,
                            is_cancelled=job.cancel_event.is_set)
        data = api.request_texture(request["image_png"], request["prompt"],
                                   request["face_size"], request["faces"])
    except client.Cancelled:
        return
    except client.ServerError as err:
        if not job.cancel_event.is_set():
            job.finish("failed", str(err))
        return
    except Exception as err:                       # a bug: still tell the user
        if not job.cancel_event.is_set():
            job.finish("failed", f"Unexpected error: {err!r}")
        return
    if not job.cancel_event.is_set():
        job.result = data
        job.state = "received"
        job.message = "Applying texture…"


def _tick():
    """Runs on the main thread every half second while a job is active."""
    job = _job
    _redraw()
    if job is None or not job.running:
        return None                                # stop the timer
    if job.state == "received":
        try:
            done = _apply(job)
        except Exception as err:
            job.finish("failed", f"Couldn't apply the result: {err}")
            done = True
        if done:
            _redraw()
            return None
    return 0.5


def _apply(job):
    """Load the returned image and project it. Returns False to retry later."""
    obj = bpy.data.objects.get(job.object_name)
    if obj is None or obj.type != "MESH":
        job.finish("failed", f"'{job.object_name}' was deleted or renamed while waiting")
        return True
    if obj.mode == "EDIT":
        # Mesh data written now would be overwritten when leaving Edit Mode.
        job.message = "Result ready: leave Edit Mode to apply it"
        return False

    ext = ".jpg" if job.result[:3] == b"\xff\xd8\xff" else ".png"
    fd, path = tempfile.mkstemp(prefix="ai_texture_result_", suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(job.result)
    image = bpy.data.images.load(path)
    image.name = f"AI Texture {obj.name}"
    width, height = image.size
    if not width:
        bpy.data.images.remove(image)
        job.finish("failed", "The server's reply couldn't be opened as an image")
        return True
    layout = projection.detect_layout(width, height)
    if layout is None:
        bpy.data.images.remove(image)
        job.finish("failed", f"The server returned a {width}×{height} image; expected a "
                             "4:3 cross (or another cubemap layout)")
        return True
    image.pack()                                   # keep it inside the .blend
    try:
        os.remove(path)
    except OSError:
        pass

    opts = job.options
    material = texturing.ensure_material(image, opts["shading"], key=obj,
                                         name=f"AI Texture {obj.name}")
    error = texturing.apply_cubemap(
        obj, image, material, layout=layout, mode="BOX",
        front_axis=opts["front_axis"], uniform=opts["uniform"],
        padding=opts["padding"], bounds=job.bounds)
    if error:
        job.finish("failed", error)
        return True

    scene = bpy.data.scenes.get(job.scene_name)
    if scene is not None:
        scene.ai_texture.last_result = image
    _undo_push("AI Texture")
    job.finish("done", f"Textured '{obj.name}' in {job.elapsed:.0f} s")
    return True


def _undo_push(message):
    """Make the change undoable as one step. Timers have no window context,
    so borrow the first window; harmless if it fails (e.g. in background mode)."""
    wm = bpy.context.window_manager
    if not wm or not wm.windows:
        return
    try:
        with bpy.context.temp_override(window=wm.windows[0]):
            bpy.ops.ed.undo_push(message=message)
    except Exception:
        pass


def _redraw():
    wm = bpy.context.window_manager
    if not wm:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type in {"PROPERTIES", "VIEW_3D"}:
                area.tag_redraw()


def shutdown():
    """Called on unregister: stop watching, let the thread die quietly."""
    global _job
    if _job and _job.running:
        _job.cancel_event.set()
    _job = None
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
