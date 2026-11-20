"""Render the object from six sides and assemble the views into one image.

Each view is an orthographic render looking straight at one side of the
object's bounding box, framed to exactly that side. Because the projection
uses the same box (texturing.object_bounds), pixel (u, v) of every view maps
back to the same spot on the surface it came from.

Rendering happens in a temporary scene that contains only the object and a
camera, so the user's scene, camera, lights and settings are never touched.
"""

import os
import struct
import tempfile
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import projection, texturing

LAYOUT = "CROSS_H"
RENDER_ENGINE = "BLENDER_WORKBENCH"    # fast, needs no lights, looks like the viewport


def _setup_scene(scene, size, style):
    r = scene.render
    r.engine = RENDER_ENGINE
    r.resolution_x = r.resolution_y = size
    r.resolution_percentage = 100
    r.film_transparent = True                       # background = alpha 0
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"  # no Filmic/AgX tone change
    scene.view_settings.look = "None"

    disp = scene.display
    disp.render_aa = "8"
    shading = disp.shading
    if style == "TEXTURED":
        shading.light = "FLAT"                      # true colours, no baked-in shading
        shading.color_type = "TEXTURE"              # current materials & textures
        shading.show_cavity = False
    else:
        shading.light = "STUDIO"
        shading.color_type = "SINGLE"               # neutral clay
        shading.single_color = (0.8, 0.8, 0.8)
        shading.show_cavity = True                  # darkens creases: shows form
        shading.cavity_type = "BOTH"


def _camera_for_side(obj, mid, half, side, front_axis):
    """World matrix, ortho size, pixel aspect and mirror flag for one view."""
    rot_t = projection.FRONT_AXIS_MATRICES[front_axis].T      # projection -> object
    n, right, up = (np.array(v, dtype=np.float64) for v in projection.SIDE_FRAMES[side])
    i_n, i_r, i_u = (int(np.abs(v).argmax()) for v in (n, right, up))

    mw = obj.matrix_world
    m3 = mw.to_3x3()
    n_w = m3 @ Vector(rot_t @ n)            # scaled by the object's scale
    r_w = m3 @ Vector(rot_t @ right)
    u_w = m3 @ Vector(rot_t @ up)
    half_n, half_w, half_h = half[i_n] * n_w.length, half[i_r] * r_w.length, half[i_u] * u_w.length
    n_w.normalize()
    r_w.normalize()
    u_w.normalize()

    # A right-handed camera frame (x = right, y = up, z = towards the viewer).
    # With a negative object scale the true frame is mirrored; render the
    # unmirrored one and flip the picture afterwards.
    cam_y = (u_w - n_w * u_w.dot(n_w)).normalized()
    cam_x = cam_y.cross(n_w)
    mirror = cam_x.dot(r_w) < 0

    face_center = mw @ Vector(rot_t @ (mid + n * half[i_n]))
    margin = max(half_n, half_w, half_h) + 0.1
    location = face_center + n_w * margin

    basis = Matrix((cam_x, cam_y, n_w)).transposed()
    matrix = basis.to_4x4()
    matrix.translation = location
    clip_end = margin + 2.0 * half_n + margin
    return matrix, 2.0 * max(half_w, half_h), half_w / half_h, mirror, margin, clip_end


def _world_extents(obj, half, front_axis):
    """Size of the bounding box along projection X (width), Y (depth) and Z
    (height), in world units, so the object's scale is taken into account."""
    rot_t = projection.FRONT_AXIS_MATRICES[front_axis].T
    m3 = obj.matrix_world.to_3x3()
    return [2.0 * half[i] * (m3 @ Vector(rot_t @ np.eye(3)[i])).length for i in range(3)]


def plan_layout(obj, size, front_axis, keep_proportions):
    """Decide the capture image's layout.

    Normal: the 4:3 horizontal cross, every side stretched to a square cell.
    Keep Proportions: the bounding box unfolded at its true proportions
    (projection.proportional_net), with the same pixel budget.

    Returns (bounds, layout) where layout is a dict:
        name         "horizontal_cross" or "proportional_cross"
        size         (width, height) of the image in pixels
        pixel_rects  {side: (x, y, w, h)} top-left origin
        uv           what projection.compute_uvs takes as `layout`
    """
    # Both layouts map each side onto its whole cell, so the projection uses
    # the true box (uniform=False); proportions come from the cell sizes.
    bounds = texturing.object_bounds(obj, front_axis, uniform=False)
    if not keep_proportions:
        width, height, rects = projection.grid_pixel_rects(LAYOUT, size)
        return bounds, {"name": "horizontal_cross", "size": (width, height),
                        "pixel_rects": rects, "uv": LAYOUT}
    # Size the cells from the true extents (flat sides stay thin).
    _mid, true_half = texturing.object_bounds(obj, front_axis, uniform=False, flat_fix=False)
    ex, ey, ez = _world_extents(obj, true_half, front_axis)
    width, height, rects = projection.proportional_net(
        ex, ey, ez, budget_px=12 * size * size, max_side=min(8 * size, 8192))
    return bounds, {"name": "proportional_cross", "size": (width, height),
                    "pixel_rects": rects,
                    "uv": projection.net_uv_rects(width, height, rects)}


def _aim(scene, cam, cam_data, obj, mid, half, side, front_axis, w, h):
    """Point the capture camera at one side, rendering w x h pixels.
    Returns True if the picture must be mirrored afterwards."""
    matrix, ortho, aspect, mirror, clip_start, clip_end = \
        _camera_for_side(obj, mid, half, side, front_axis)
    cam.matrix_world = matrix
    cam_data.ortho_scale = ortho
    cam_data.clip_start = clip_start * 0.5
    cam_data.clip_end = clip_end
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    # The side's real shape is `aspect` (width/height); the cell is w x h
    # pixels. Non-square pixels make up the difference, exactly like the
    # projection stretches it back. In the proportional layout the cell
    # already has the side's shape, so this is ~1.
    pixel = min(max(aspect / (w / h), 1 / 200), 200)
    scene.render.pixel_aspect_x = max(pixel, 1.0)
    scene.render.pixel_aspect_y = max(1.0 / pixel, 1.0)
    return mirror


def _render_pixels(scene, path, w, h, mirror):
    """Render the capture scene to `path` and return (h, w, 4) floats, top row first."""
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=scene.name)
    img = bpy.data.images.load(path)
    try:
        pixels = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(img)
    view = pixels.reshape(h, w, 4)[::-1]           # Blender stores bottom row first
    return view[:, ::-1] if mirror else view


LOOKUP_ATTR = "_uvusee_lookup"
LOOKUP_MAX_SIDE = 2048       # lookup images: up to 2x the view, at most this many pixels


def _render_lookups(scene, cam, cam_data, obj, mid, half, front_axis, layout, uv_name, tmpdir):
    """For Blend & Bake: render each view again as a lookup image whose pixels
    hold (u, v, cosine, coverage) of the surface seen there.

    A temporary float colour attribute carries u, v (from the bake UV map) and
    the cosine between the surface normal and this view's direction; Workbench
    draws it unlit, without anti-aliasing, into a 32-bit EXR, so the values come
    back exactly. The renderer's depth test decides which surface is visible.
    """
    mesh = obj.data
    nloops = len(mesh.loops)
    uv = np.empty(nloops * 2, dtype=np.float32)
    mesh.uv_layers[uv_name].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    # Smooth (vertex) normals, not the shading normals: on flat-shaded meshes the
    # latter are constant per face, which would make the blend jump at every edge.
    vnormals = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertex_normals.foreach_get("vector", vnormals)
    loop_verts = np.empty(nloops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    normals = vnormals.reshape(-1, 3)[loop_verts]
    m3 = np.array(obj.matrix_world.to_3x3(), dtype=np.float64)
    # Normals transform with the inverse transpose (correct for any scale).
    world_n = normals @ np.linalg.inv(m3)
    world_n /= np.maximum(np.linalg.norm(world_n, axis=1, keepdims=True), 1e-12)
    rot_t = projection.FRONT_AXIS_MATRICES[front_axis].T

    previous_active = mesh.color_attributes.active_color_name
    attr = mesh.color_attributes.new(LOOKUP_ATTR, "FLOAT_COLOR", "CORNER")
    data = np.ones((nloops, 4), dtype=np.float32)
    data[:, 0:2] = uv

    r = scene.render
    r.image_settings.file_format = "OPEN_EXR"
    r.image_settings.color_depth = "32"
    r.image_settings.color_mode = "RGBA"
    r.dither_intensity = 0.0
    scene.display.render_aa = "OFF"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "VERTEX"
    shading.show_cavity = False
    shading.show_object_outline = False
    shading.show_specular_highlight = False
    shading.show_shadows = False

    lookups = {}
    try:
        mesh.color_attributes.active_color = attr
        for side in projection.SIDES:
            _x, _y, w, h = layout["pixel_rects"][side]
            scale = min(2.0, LOOKUP_MAX_SIDE / max(w, h))
            lw, lh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            view_dir = m3 @ (rot_t @ np.array(projection.SIDE_FRAMES[side][0], dtype=np.float64))
            view_dir /= np.linalg.norm(view_dir)
            data[:, 2] = world_n @ view_dir
            attr.data.foreach_set("color", data.ravel())
            mesh.update()
            mirror = _aim(scene, cam, cam_data, obj, mid, half, side, front_axis, lw, lh)
            lookups[side] = _render_pixels(scene, os.path.join(tmpdir, f"lookup_{side.lower()}.exr"),
                                           lw, lh, mirror).copy()
    finally:
        mesh.color_attributes.remove(mesh.color_attributes[LOOKUP_ATTR])
        if previous_active and previous_active in mesh.color_attributes:
            mesh.color_attributes.active_color = mesh.color_attributes[previous_active]
        mesh.update()
    return lookups


def capture_views(obj, size, front_axis, uniform, style, progress=None, bake_uv=None):
    """Render six views of `obj` and assemble them into one image: the 4:3
    horizontal cross, or with `uniform` (Keep Proportions) the object's box
    unfolded at its true proportions.

    bake_uv: name of a UV map. If given, lookup images for Blend & Bake are
    rendered too and returned in layout["lookups"] (see bake.py).

    Returns (blender_image, png_bytes, bounds, layout); see plan_layout. The
    image is packed into the .blend so the thumbnail survives saving.
    """
    if obj.mode == "EDIT":
        obj.update_from_editmode()
    bounds, layout = plan_layout(obj, size, front_axis, uniform)
    mid, half = bounds
    width, height = layout["size"]

    tmpdir = tempfile.mkdtemp(prefix="ai_texture_")
    scene = bpy.data.scenes.new("AI Texture Capture")
    cam_data = bpy.data.cameras.new("AI Texture Capture")
    cam = bpy.data.objects.new("AI Texture Capture", cam_data)
    sheet = np.zeros((height, width, 4), dtype=np.float32)   # top row first
    was_hidden = obj.hide_render

    try:
        obj.hide_render = False
        _setup_scene(scene, size, style)
        cam_data.type = "ORTHO"
        cam_data.sensor_fit = "AUTO"
        scene.collection.objects.link(cam)
        scene.collection.objects.link(obj)          # the same object, not a copy
        scene.camera = cam

        for step, side in enumerate(projection.SIDES):
            x, y, w, h = layout["pixel_rects"][side]
            mirror = _aim(scene, cam, cam_data, obj, mid, half, side, front_axis, w, h)
            sheet[y:y + h, x:x + w] = _render_pixels(
                scene, os.path.join(tmpdir, f"{side.lower()}.png"), w, h, mirror)
            if progress:
                progress(step + 1)
        if bake_uv:
            layout["lookups"] = _render_lookups(scene, cam, cam_data, obj, mid, half,
                                                front_axis, layout, bake_uv, tmpdir)
    finally:
        obj.hide_render = was_hidden
        bpy.data.scenes.remove(scene)
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_data)

    out_path = os.path.join(tmpdir, "cubemap.png")
    image = bpy.data.images.new(f"AI Capture {obj.name}", width, height, alpha=True)
    image.pixels.foreach_set(sheet[::-1].ravel())
    image.filepath_raw = out_path
    image.file_format = "PNG"
    image.save()
    with open(out_path, "rb") as f:
        png_bytes = f.read()
    image.pack()
    return image, png_bytes, bounds, layout


def flatten_png(image, color=(0.0, 0.0, 0.0), canvas=None, out_size=None):
    """PNG bytes of `image` laid over a plain background (no transparency),
    for hosted models (ChatGPT / Nano Banana).

    Hosted image models tend to treat transparent areas unpredictably, so they
    get the views on solid black instead (the prompt tells them so).
    canvas = (width, height, x, y): place the image at (x, y), top-left origin,
    on a larger canvas of that size, e.g. to reach an aspect ratio the model
    supports. jobs.py crops the same rectangle out of the answer.
    out_size = (width, height): scale the canvas to this size before encoding
    (the size the model will answer at; no point uploading more pixels).

    The PNG is written by encode_png, not by Blender, so it holds only the
    pixels: Blender's writer adds colour-profile, EXIF and offset blocks that
    image APIs don't need and may refuse.
    """
    src_w, src_h = image.size
    src = np.empty(src_w * src_h * 4, dtype=np.float32)
    image.pixels.foreach_get(src)
    src = src.reshape(src_h, src_w, 4)[::-1]                   # top row first
    alpha = src[..., 3:4]
    rgb = src[..., :3] * alpha + np.array(color, dtype=np.float32) * (1.0 - alpha)

    width, height, ox, oy = canvas or (src_w, src_h, 0, 0)
    out = np.empty((height, width, 3), dtype=np.float32)
    out[...] = color
    out[oy:oy + src_h, ox:ox + src_w] = rgb

    if out_size and tuple(out_size) != (width, height):
        out = _resize(out, *out_size)
    return encode_png(out)


def _resize(rgb, width, height):
    """Scale an (h, w, 3) float image with Blender's image scaler."""
    src_h, src_w = rgb.shape[:2]
    tmp = bpy.data.images.new("AI Capture resize", src_w, src_h, alpha=False, float_buffer=True)
    try:
        rgba = np.ones((src_h, src_w, 4), dtype=np.float32)
        rgba[..., :3] = rgb[::-1]                              # Blender: bottom row first
        tmp.pixels.foreach_set(rgba.ravel())
        tmp.scale(width, height)
        scaled = np.empty(width * height * 4, dtype=np.float32)
        tmp.pixels.foreach_get(scaled)
    finally:
        bpy.data.images.remove(tmp)
    return scaled.reshape(height, width, 4)[::-1, :, :3]


def encode_png(rgb):
    """A minimal standard PNG (8-bit RGB, no metadata) from an (h, w, 3) float
    image with values 0..1, top row first."""
    height, width = rgb.shape[:2]
    data = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    raw = np.zeros((height, 1 + width * 3), dtype=np.uint8)   # filter byte 0 per row
    raw[:, 1:] = data.reshape(height, width * 3)

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)   # 8-bit, RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def faces_description(layout):
    """Where each side sits in the sent image, for the server.

    horizontal_cross:   {"front": [col, row], ...}       (grid cells, as before)
    proportional_cross: {"front": [x, y, w, h], ...}     (pixels, top-left origin)
    """
    if layout["name"] == "horizontal_cross":
        _cols, _rows, cells = projection.LAYOUTS[LAYOUT]
        return {side.lower(): [col, row] for side, (col, row, _t) in cells.items()}
    return face_rects(layout)


def face_rects(layout):
    """{"front": [x, y, w, h], ...} in pixels, top-left origin, for any layout."""
    return {side.lower(): list(rect) for side, rect in layout["pixel_rects"].items()}