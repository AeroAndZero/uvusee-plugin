"""One AI request at a time: a background thread does the networking, and a
timer on Blender's main thread watches it and applies the result.

Why the split: waiting for a server can take minutes, and Blender's UI would
freeze if we waited in an operator. But Blender's data may only be changed
from the main thread, so the thread never touches bpy; it just fills in
fields on the Job, and the timer (which runs on the main thread) acts on them.
"""

import json
import os
import tempfile
import threading
import time

import bpy
import numpy as np

from . import bake, capture, client, projection, providers, texturing

ACTIVE = {"uploading", "waiting", "received", "baking", "baked"}


def _clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class Job:
    def __init__(self, object_name, scene_name, bounds, options, provider_label=""):
        self.object_name = object_name
        self.provider_label = provider_label   # "ChatGPT" / "Nano Banana", "" = custom server
        self.scene_name = scene_name
        self.bounds = bounds          # the capture's bounding box, reused when applying
        self.options = options        # projection settings at the time of sending
        self.state = "uploading"      # uploading / waiting / received / done / failed / cancelled
        self.message = ""             # final message, or the Edit Mode hint
        self.result = None
        self.started = time.monotonic()
        self.finished = None
        self.cancel_event = threading.Event()

        # Live status reported by the server (written by the worker thread).
        self.phase = "uploading"      # uploading / queued / running / downloading / retrying
        self.phase_started = self.started
        self.position = None          # jobs ahead of ours in the server's queue
        self.progress = None          # 0..1 if the server reports it
        self.server_message = ""
        self.retry = None             # (attempt, max_attempts) while reconnecting
        self.last_contact = None      # when the server last answered

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

    def update(self, info):
        """Called from the worker thread with what the client just learned."""
        if not self.running:
            return
        now = time.monotonic()
        phase = info["phase"]
        if phase != self.phase:
            self.phase_started = now
        self.phase = phase
        self.state = "waiting"
        if phase == "retrying":
            self.retry = (info.get("attempt"), info.get("max_attempts"))
            self.server_message = info.get("message", "")
            return
        self.retry = None
        self.position = info.get("position")
        self.progress = info.get("progress")
        self.server_message = info.get("message", "")
        if phase != "uploading" and info.get("polled", True):
            self.last_contact = now

    def describe(self):
        """(icon, headline, detail) for the panel."""
        now = time.monotonic()
        if self.state == "received":
            return "BRUSH_DATA", self.message or "Applying texture…", ""
        if self.state in ("baking", "baked"):
            return "TEXTURE", self.message or "Baking the texture…", \
                f"Total {_clock(time.monotonic() - self.started)}"
        phase = self.phase
        if phase == "uploading":
            icon, head = "EXPORT", "Uploading views to the server…"
        elif phase == "running" and self.provider_label:
            # Hosted models answer in one go: no queue, no progress, just a timer.
            icon, head = "SHADERFX", f"Generating with {self.server_message or self.provider_label}…"
        elif phase == "queued":
            icon = "SORTTIME"
            if self.position is None:
                head = "Queued on the server"
            elif self.position == 0:
                head = "Queued: next in line"
            else:
                head = f"Queued: {self.position} job{'s' if self.position != 1 else ''} ahead"
        elif phase == "running":
            icon, head = "SHADERFX", "Generating on the server"
            if self.server_message:
                head += f": {self.server_message}"
        elif phase == "downloading":
            icon, head = "IMPORT", "Downloading the result…"
        elif phase == "retrying":
            attempt, most = self.retry or (None, None)
            icon = "ERROR"
            head = (f"Retrying ({attempt}/{most})" if self.provider_label
                    else f"Lost contact, retrying ({attempt}/{most})")
            if self.server_message:
                head += f": {self.server_message}"
        else:
            icon, head = "SORTTIME", phase

        detail = f"Total {_clock(now - self.started)}"
        if phase in {"queued", "running"}:
            detail += f" · this step {_clock(now - self.phase_started)}"
        if self.last_contact is not None and phase != "uploading":
            detail += f" · checked {now - self.last_contact:.0f} s ago"
        return icon, head, detail


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
    if request.get("provider", "CUSTOM") == "CUSTOM":
        api = client.Client(request["server_url"], request["api_key"],
                            timeout=request["timeout"],
                            poll_interval=request["poll_interval"],
                            max_run_time=request.get("max_run_time", 0),
                            on_status=job.update,
                            is_cancelled=job.cancel_event.is_set)
        call = lambda: api.request_texture(  # noqa: E731
            request["image_png"], request["prompt"], request["face_size"], request["faces"],
            layout=request["layout_name"],
            extra_fields={"face_rects": json.dumps(request["face_rects"])})
    else:
        api = providers.HostedClient(request["provider"], request["api_key"], request["model"],
                                     timeout=request["timeout"], quality=request["quality"],
                                     upload_copy=request.get("upload_copy", ""),
                                     on_status=job.update,
                                     is_cancelled=job.cancel_event.is_set)
        call = lambda: api.generate(request["image_png"], request["prompt"],  # noqa: E731
                                    request["plan"])
    try:
        data = call()
    except client.Cancelled:
        api.cancel_remote()          # lets the server drop the job, if it supports that
        return
    except client.ServerError as err:
        if not job.cancel_event.is_set():
            job.finish("failed", str(err))
        return
    except Exception as err:                       # a bug: still tell the user
        if not job.cancel_event.is_set():
            job.finish("failed", f"Unexpected error: {err!r}")
        return
    if job.cancel_event.is_set():
        return
    job.result = data
    job.message = "Applying texture…"
    job.state = "received"


def _tick():
    """Runs on the main thread every half second while a job is active."""
    job = _job
    _redraw()
    if job is None or not job.running:
        return None                                # stop the timer
    if job.state in ("received", "baked"):
        try:
            done = _apply(job) if job.state == "received" else _finish_bake(job)
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
    opts = job.options
    bake_opts = opts.get("bake")
    sent = opts["layout"]                          # the layout of the capture we sent
    cap_w, cap_h = sent["size"]
    who = job.provider_label or "The server"
    canvas = opts.get("canvas")

    def shape_error(expected_w, expected_h):
        bpy.data.images.remove(image)
        job.finish("failed", f"{who} returned a {width}×{height} image, but a picture shaped "
                             f"like the {expected_w}×{expected_h} one it was sent is needed")
        return True

    if canvas:
        # Hosted model: it answered at its own resolution for the (possibly
        # padded) canvas we sent. Scale back to the canvas, crop the capture out.
        cw, ch, ox, oy = canvas
        if abs((width / height) / (cw / ch) - 1.0) > 0.06:
            return shape_error(cw, ch)
        if (width, height) != (cw, ch):
            image.scale(cw, ch)
        if (cw, ch) != (cap_w, cap_h):
            image = _crop(image, ox, oy, cap_w, cap_h)
        width, height = cap_w, cap_h
        layout = sent["uv"]
    elif sent["name"] == "proportional_cross" or bake_opts:
        # Custom server with the Keep Proportions layout, or any layout when
        # baking: any resolution with the same shape works; scale it to the
        # capture's size so it lines up with the lookup images.
        if abs((width / height) / (cap_w / cap_h) - 1.0) > 0.06:
            return shape_error(cap_w, cap_h)
        if (width, height) != (cap_w, cap_h):
            image.scale(cap_w, cap_h)
            width, height = cap_w, cap_h
        layout = sent["uv"]
    else:
        # Custom server with the 4:3 cross: any cubemap layout is accepted.
        layout = projection.detect_layout(width, height)
        if layout is None:
            bpy.data.images.remove(image)
            job.finish("failed", f"The server returned a {width}×{height} image; expected a "
                                 "4:3 cross (or another cubemap layout)")
            return True
    image.name = f"AI Texture {obj.name}"
    image.pack()                                   # keep it inside the .blend
    try:
        os.remove(path)
    except OSError:
        pass

    if bake_opts:
        # Blend & Bake: the heavy lifting is numpy, done on a background thread
        # so Blender stays responsive; _finish_bake picks the result up.
        image.name = f"AI Result {obj.name}"
        job.raw_image_name = image.name
        painted = _pixels_rgb(image)
        views = [(sent["pixel_rects"][side], sent["lookups"][side]) for side in projection.SIDES]
        job.state = "baking"
        job.message = "Baking the texture…"
        threading.Thread(target=_bake_worker, args=(job, painted, views, bake_opts),
                         daemon=True).start()
        return False

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


def _pixels_rgb(image):
    """(H, W, 3) float RGB of a Blender image, top row first."""
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(height, width, 4)[::-1, :, :3].copy()


def _bake_worker(job, painted, views, opts):
    """Background thread: pure numpy, never touches bpy."""
    try:
        texture, _coverage = bake.bake(painted, views, size=opts["size"],
                                       sharpness=opts["sharpness"], min_cos=opts["min_cos"])
        png = capture.encode_png(texture)
    except Exception as err:
        if job.running:
            job.finish("failed", f"Baking failed: {err!r}")
        return
    if job.running:
        job.baked_png = png
        job.state = "baked"


def _finish_bake(job):
    """Main thread: turn the baked pixels into an image and material."""
    obj = bpy.data.objects.get(job.object_name)
    if obj is None or obj.type != "MESH":
        job.finish("failed", f"'{job.object_name}' was deleted or renamed while waiting")
        return True
    if obj.mode == "EDIT":
        job.message = "Texture baked: leave Edit Mode to apply it"
        return False
    opts = job.options
    uv_name = opts["bake"]["uv"]

    fd, path = tempfile.mkstemp(prefix="ai_texture_baked_", suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(job.baked_png)
    texture = bpy.data.images.load(path)
    texture.name = f"AI Texture {obj.name}"
    texture.pack()
    try:
        os.remove(path)
    except OSError:
        pass

    material = texturing.ensure_material(texture, opts["shading"], key=obj,
                                         name=f"AI Texture {obj.name}", uv_name=uv_name)
    error = texturing.apply_baked(obj, material, uv_name)
    if error:
        job.finish("failed", error)
        return True
    scene = bpy.data.scenes.get(job.scene_name)
    raw = bpy.data.images.get(getattr(job, "raw_image_name", ""))
    if scene is not None and raw is not None:
        scene.ai_texture.last_result = raw        # the "Received" thumbnail: what came back
    _undo_push("AI Texture")
    job.finish("done", f"Textured '{obj.name}' in {job.elapsed:.0f} s")
    return True


def _crop(image, x, y, width, height):
    """A new image holding the (x, y, width, height) rectangle of `image`
    (top-left origin). Goes through a PNG file so it packs like a loaded image."""
    full_w, full_h = image.size
    pixels = np.empty(full_w * full_h * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(full_h, full_w, 4)                 # bottom row first
    bottom = full_h - (y + height)
    part = np.ascontiguousarray(pixels[bottom:bottom + height, x:x + width])

    fd, path = tempfile.mkstemp(prefix="ai_texture_crop_", suffix=".png")
    os.close(fd)
    tmp = bpy.data.images.new("AI Texture crop", width, height, alpha=True)
    try:
        tmp.pixels.foreach_set(part.ravel())
        tmp.filepath_raw = path
        tmp.file_format = "PNG"
        tmp.save()
    finally:
        bpy.data.images.remove(tmp)
    cropped = bpy.data.images.load(path)
    cropped.pack()
    bpy.data.images.remove(image)
    try:
        os.remove(path)
    except OSError:
        pass
    return cropped


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