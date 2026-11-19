"""Render the object from six sides and assemble the views into one image.

Each view is an orthographic render looking straight at one side of the
object's bounding box, framed to exactly that side. Because the projection
uses the same box (texturing.object_bounds), pixel (u, v) of every view maps
back to the same spot on the surface it came from.

Rendering happens in a temporary scene that contains only the object and a
camera, so the user's scene, camera, lights and settings are never touched.
"""

import os
import tempfile

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


def capture_views(obj, size, front_axis, uniform, style, progress=None):
    """Render six views of `obj` and assemble them as a horizontal cross.

    Returns (blender_image, png_bytes, bounds). The image is packed into the
    .blend so the thumbnail survives saving.
    """
    if obj.mode == "EDIT":
        obj.update_from_editmode()
    mid, half = bounds = texturing.object_bounds(obj, front_axis, uniform)
    cols, rows, cells = projection.LAYOUTS[LAYOUT]

    tmpdir = tempfile.mkdtemp(prefix="ai_texture_")
    scene = bpy.data.scenes.new("AI Texture Capture")
    cam_data = bpy.data.cameras.new("AI Texture Capture")
    cam = bpy.data.objects.new("AI Texture Capture", cam_data)
    sheet = np.zeros((rows * size, cols * size, 4), dtype=np.float32)   # top row first
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
            matrix, ortho, aspect, mirror, clip_start, clip_end = \
                _camera_for_side(obj, mid, half, side, front_axis)
            cam.matrix_world = matrix
            cam_data.ortho_scale = ortho
            cam_data.clip_start = clip_start * 0.5
            cam_data.clip_end = clip_end
            # Non-square pixels squeeze a non-square side into a square cell,
            # exactly like the projection stretches it back.
            aspect = min(max(aspect, 1 / 200), 200)
            scene.render.pixel_aspect_x = max(aspect, 1.0)
            scene.render.pixel_aspect_y = max(1.0 / aspect, 1.0)

            path = os.path.join(tmpdir, f"{side.lower()}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True, scene=scene.name)

            view_img = bpy.data.images.load(path)
            try:
                pixels = np.empty(size * size * 4, dtype=np.float32)
                view_img.pixels.foreach_get(pixels)
            finally:
                bpy.data.images.remove(view_img)
            view = pixels.reshape(size, size, 4)[::-1]       # Blender stores bottom row first
            if mirror:
                view = view[:, ::-1]
            col, row, _turns = cells[side]
            sheet[row * size:(row + 1) * size, col * size:(col + 1) * size] = view
            if progress:
                progress(step + 1)
    finally:
        obj.hide_render = was_hidden
        bpy.data.scenes.remove(scene)
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_data)

    out_path = os.path.join(tmpdir, "cubemap.png")
    image = bpy.data.images.new(f"AI Capture {obj.name}", cols * size, rows * size, alpha=True)
    image.pixels.foreach_set(sheet[::-1].ravel())
    image.filepath_raw = out_path
    image.file_format = "PNG"
    image.save()
    with open(out_path, "rb") as f:
        png_bytes = f.read()
    image.pack()
    return image, png_bytes, bounds


def flatten_png(image, color=(0.0, 0.0, 0.0)):
    """PNG bytes of `image` laid over a plain background (no transparency).

    Hosted image models tend to treat transparent areas unpredictably, so they
    get the views on solid black instead (the prompt tells them so).
    """
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(-1, 4)
    alpha = pixels[:, 3:4]
    pixels[:, :3] = pixels[:, :3] * alpha + np.array(color, dtype=np.float32) * (1.0 - alpha)
    pixels[:, 3] = 1.0

    path = os.path.join(tempfile.mkdtemp(prefix="ai_texture_"), "cubemap_flat.png")
    flat = bpy.data.images.new("AI Capture flat", width, height, alpha=False)
    try:
        flat.pixels.foreach_set(pixels.ravel())
        flat.filepath_raw = path
        flat.file_format = "PNG"
        flat.save()
    finally:
        bpy.data.images.remove(flat)
    with open(path, "rb") as f:
        data = f.read()
    try:
        os.remove(path)
    except OSError:
        pass
    return data


def faces_description():
    """Where each side sits in the sent image, for the server: {"front": [col, row]}."""
    _cols, _rows, cells = projection.LAYOUTS[LAYOUT]
    return {side.lower(): [col, row] for side, (col, row, _t) in cells.items()}
