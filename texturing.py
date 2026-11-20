"""Putting a cubemap image onto a mesh: the UV map and the material."""

import math

import bpy
import numpy as np

from . import projection

UV_NAME = "CubemapUV"         # Box projection
BAKE_UV_NAME = "UVUSeeBake"   # Blend & Bake with Auto Unwrap
KEY_PROP = "ai_texture_key"   # custom property marking materials this add-on owns


def ensure_material(image, shading, key, name, uv_name=UV_NAME):
    """Find (or make) the material identified by `key` and point it at `image`.

    `key` is any datablock: the image for manual projection (one material per
    image), the object for AI results (each new result replaces the last).
    """
    mat = next((m for m in bpy.data.materials if m.get(KEY_PROP) == key), None)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat[KEY_PROP] = key
    if mat.node_tree is None:
        mat.use_nodes = True   # Blender 4.x; materials always have nodes in 5.0+

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    uv.location = (-700, 0)

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"      # never wrap around to the opposite edge
    tex.location = (-450, 0)
    links.new(uv.outputs["UV"], tex.inputs["Vector"])

    if shading == "EMISSION":
        shader = nodes.new("ShaderNodeEmission")
        links.new(tex.outputs["Color"], shader.inputs["Color"])
    else:
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        links.new(tex.outputs["Color"], shader.inputs["Base Color"])
    shader.location = (-100, 0)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (250, 0)
    links.new(shader.outputs[0], out.inputs["Surface"])

    nodes.active = tex    # lets Solid view's "Texture" colour mode show it
    return mat


def read_mesh(mesh):
    """Copy the mesh data the projection needs into numpy arrays (fast)."""
    nv, npoly, nloop = len(mesh.vertices), len(mesh.polygons), len(mesh.loops)

    co = np.empty(nv * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    normals = np.empty(npoly * 3, dtype=np.float32)
    mesh.polygons.foreach_get("normal", normals)
    centers = np.empty(npoly * 3, dtype=np.float32)
    mesh.polygons.foreach_get("center", centers)
    loop_verts = np.empty(nloop, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    loop_start = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    loop_total = np.empty(npoly, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)

    # Which polygon each loop belongs to (polygons own consecutive loop ranges).
    order = np.argsort(loop_start)
    loop_poly = np.repeat(order, loop_total[order])

    return (co.reshape(-1, 3), normals.reshape(-1, 3), centers.reshape(-1, 3),
            loop_verts, loop_poly)


def object_bounds(obj, front_axis, uniform, flat_fix=True):
    """Bounding box used by both capture and projection.

    It covers the mesh itself AND the result of its modifiers, so e.g. a
    Mirror modifier's second half is captured, and every base vertex lands
    inside the box.
    """
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    corners = np.array([tuple(c) for c in obj.bound_box], dtype=np.float64)
    return projection.compute_bounds(np.vstack([co.reshape(-1, 3), corners]),
                                     front_axis, uniform, flat_fix)


def apply_cubemap(obj, image, material, *, layout, mode, front_axis, uniform,
                  padding, bounds=None, rotations=None, flips=None,
                  swap_front_back=False):
    """Write the CubemapUV map onto obj's mesh and assign `material` to all faces.

    Returns None on success or an error message.
    """
    mesh = obj.data
    if not mesh.polygons:
        return f"'{obj.name}' has no faces"
    width, height = image.size

    uv_layer = mesh.uv_layers.get(UV_NAME) or mesh.uv_layers.new(name=UV_NAME)
    if uv_layer is None:
        return f"'{obj.name}' already has the maximum number of UV maps"

    if bounds is None:
        bounds = object_bounds(obj, front_axis, uniform)
    uvs, _sides = projection.compute_uvs(
        *read_mesh(mesh),
        layout=layout, mode=mode, front_axis=front_axis, uniform=uniform,
        padding_px=padding, image_size=(width, height), bounds=bounds,
        rotations=rotations, flips=flips, swap_front_back=swap_front_back,
    )
    uv_layer.data.foreach_set("uv", uvs.ravel())
    mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True

    slot = mesh.materials.find(material.name)
    if slot < 0:
        mesh.materials.append(material)
        slot = len(mesh.materials) - 1
    mesh.polygons.foreach_set("material_index",
                              np.full(len(mesh.polygons), slot, dtype=np.int32))
    mesh.update()
    return None


# --- Blend & Bake ---------------------------------------------------------------

def prepare_bake_uvs(context, obj, source):
    """Make sure `obj` has the UV map the bake writes into; return its name.

    source "AUTO":   (re)unwrap with Smart UV Project into the UVUSeeBake map.
                     Other UV maps, other objects and the selection are untouched.
    source "ACTIVE": use the object's active UV map as it is (your own unwrap;
                     overlapping islands will share colours).
    Must be called in Object Mode. Raises ValueError with a message for the user.
    """
    mesh = obj.data
    if source == "ACTIVE":
        layer = mesh.uv_layers.active
        if layer is None:
            raise ValueError(f"'{obj.name}' has no UV map. Use Auto Unwrap instead")
        return layer.name

    layer = mesh.uv_layers.get(BAKE_UV_NAME) or mesh.uv_layers.new(name=BAKE_UV_NAME)
    if layer is None:
        raise ValueError(f"'{obj.name}' already has the maximum number of UV maps")
    mesh.uv_layers.active = layer

    view_layer = context.view_layer
    selected = [o for o in view_layer.objects if o.select_get()]
    active = view_layer.objects.active
    saved = {}
    for domain in ("vertices", "edges", "polygons"):
        items = getattr(mesh, domain)
        for flag in ("select", "hide"):
            values = np.empty(len(items), dtype=bool)
            items.foreach_get(flag, values)
            saved[domain, flag] = values
            # Smart UV Project only unwraps visible, selected faces.
            items.foreach_set(flag, np.full(len(items), flag == "select"))
    try:
        for o in selected:
            o.select_set(False)
        obj.select_set(True)
        view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.006,
                                 correct_aspect=True, scale_to_bounds=False)
    finally:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for (domain, flag), values in saved.items():
            getattr(mesh, domain).foreach_set(flag, values)
        mesh.update()
        obj.select_set(False)
        for o in selected:
            o.select_set(True)
        view_layer.objects.active = active
    return BAKE_UV_NAME


def apply_baked(obj, material, uv_name):
    """Show the baked texture: make `uv_name` the active/render UV map and put
    `material` on every face (other slots stay, unused)."""
    mesh = obj.data
    layer = mesh.uv_layers.get(uv_name)
    if layer is None:
        return f"'{obj.name}' no longer has the UV map '{uv_name}'"
    mesh.uv_layers.active = layer
    layer.active_render = True
    slot = mesh.materials.find(material.name)
    if slot < 0:
        mesh.materials.append(material)
        slot = len(mesh.materials) - 1
    mesh.polygons.foreach_set("material_index",
                              np.full(len(mesh.polygons), slot, dtype=np.int32))
    mesh.update()
    return None